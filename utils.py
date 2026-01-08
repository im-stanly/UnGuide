import sys
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np
from ldm.models.diffusion.ddimcopy import DDIMSampler
from diffusers import FluxPipeline, AutoencoderTiny
from diffusers.models import FluxTransformer2DModel
from diffusers.utils import make_image_grid
from peft import LoraConfig, get_peft_model

sys.path.append('.')
from utils.flux_utils import esd_flux_call
FluxPipeline.__call__ = esd_flux_call

def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def load_model_from_config(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Load and initialize model from configuration and checkpoint"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    # Load checkpoint weights
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(pl_sd["state_dict"], strict=False)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device

    return model

def get_models(config_path: str, ckpt_path: str, device: str):
    """Initialize both original and trainable models with samplers"""
    # Original model (frozen, for reference)
    model_orig = load_model_from_config(config_path, ckpt_path, device)
    sampler_orig = DDIMSampler(model_orig)

    # Trainable model (will be modified with LoRA)
    model = load_model_from_config(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def load_flux_models(torch_dtype=torch.bfloat16, device='cuda:0', lora_rank=16):
    basemodel_id="black-forest-labs/FLUX.1-dev"
    
    esd_transformer = FluxTransformer2DModel.from_pretrained(basemodel_id, subfolder="transformer", torch_dtype=torch_dtype).to(device)
    pipe_orig = FluxPipeline.from_pretrained(basemodel_id,
                                        transformer=esd_transformer,
                                        vae=None,
                                        torch_dtype=torch_dtype, 
                                        use_safetensors=True).to(device)

    pipe = FluxPipeline.from_pretrained(basemodel_id,
                                             transformer=esd_transformer,
                                             vae=None,
                                             torch_dtype=torch_dtype,
                                             use_safetensors=True).to(device)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.train()
    sampler = DDIMSampler(pipe)

    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(device)
    pipe_orig.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(device)

    return pipe, pipe_orig, sampler

def print_trainable_parameters(model, max_params: int = 10):
    """Print the first few trainable parameters"""
    print(f"First {max_params} layers with requires_grad == True:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"  {name}")
            count += 1
            if count >= max_params:
                break

def apply_lora_to_model(model, lora_state_dict, alpha=4):
    """
    Apply LoRA adapters to a base model’s weights, scaling by the given alpha.

    Args:
        model (nn.Module): The base model containing original weights.
        lora_state_dict (dict): A state dict containing keys like
            "<prefix>.lora.A" and "<prefix>.lora.B" for each adapter.
        alpha (float): Scaling factor for the LoRA update (default: 4).
    """
    model_sd = model.state_dict()

    for lora_A_key in [k for k in lora_state_dict if k.endswith(".lora.A")]:
        prefix = lora_A_key[:-len(".lora.A")]

        A_key = prefix + ".lora.A"
        B_key = prefix + ".lora.B"
        W_key = prefix + ".weight"  # the original weight in the model


        A = lora_state_dict[A_key].to(model_sd[W_key].device)
        B = lora_state_dict[B_key].to(model_sd[W_key].device)

        delta = A.matmul(B)
        delta = delta.T
        
        model_sd[W_key] = model_sd[W_key] + alpha * delta

    model.load_state_dict(model_sd, strict=False)
