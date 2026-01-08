import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from torchvision.transforms.functional import to_pil_image
import copy
import random

from diffusers import FluxPipeline
from utils import set_seed, apply_lora_to_model

import sys
sys.path.append('.')
from utils.flux_utils import esd_flux_call
FluxPipeline.__call__ = esd_flux_call


def load_flux_models(basemodel_id="black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device='cuda:0'):
    """Load Flux models for generation"""
    from diffusers import AutoencoderTiny
    from diffusers.models import FluxTransformer2DModel
    
    esd_transformer = FluxTransformer2DModel.from_pretrained(
        basemodel_id, subfolder="transformer", torch_dtype=torch_dtype
    ).to(device)
    
    pipe_orig = FluxPipeline.from_pretrained(
        basemodel_id,
        transformer=esd_transformer,
        vae=None,
        torch_dtype=torch_dtype,
        use_safetensors=True
    ).to(device)

    pipe = FluxPipeline.from_pretrained(
        basemodel_id,
        transformer=esd_transformer,
        vae=None,
        torch_dtype=torch_dtype,
        use_safetensors=True
    ).to(device)

    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(device)
    pipe_orig.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch_dtype).to(device)

    pipe.set_progress_bar_config(disable=True)
    pipe_orig.set_progress_bar_config(disable=True)

    return pipe, pipe_orig


def compute_latent_diff(
    prompt,
    model,
    model_orig,
    guidance_scale,
    seed,
    repeats=30,
    batch_size=1,
    image_size=512,
    run_till_timestep=20,
    num_inference_steps=28,
    max_sequence_length=512,
    device="cuda",
    torch_dtype=torch.bfloat16,
):
    """Compute latent difference between original and modified Flux models"""
    diffs_all = []

    set_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    height = width = image_size
    prompts = [prompt] * batch_size
    
    with torch.no_grad():
        # Encode prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = model.encode_prompt(
            prompts, prompt_2=prompts, max_sequence_length=max_sequence_length
        )

        # Generate intermediate latent
        xt = model(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            run_till_timestep=run_till_timestep,
            generator=gen,
            output_type='latent',
            height=height,
            width=width,
        ).images

        # Prepare inputs for transformer
        timesteps = model.scheduler.timesteps[run_till_timestep].unsqueeze(0).to(device)
        guidance = torch.tensor([1.0], device=device).expand(batch_size)
        
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            xt.shape[0],
            xt.shape[2] // 2,
            xt.shape[3] // 2,
            device,
            torch_dtype,
        )

        # Get predictions from both models
        eps_lora = model.transformer(
            hidden_states=xt,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        eps_orig = model_orig.transformer(
            hidden_states=xt,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        diffs = (
            (eps_lora - eps_orig)
            .view(batch_size, -1)
            .norm(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )

    return diffs


def decide_w(prompt_diff, empty_diff, w1=-1.0, w2=2.0):
    """Decide dynamic guidance weight based on latent differences"""
    prompt_diff = np.mean(prompt_diff)
    empty_diff = np.mean(empty_diff)
    
    if prompt_diff > empty_diff:
        return w1
    else:
        return w2


class AutoGuidedFluxModel:
    """Wrapper for Flux model with automatic guidance weight"""
    def __init__(self, model_full, model_unlearned, w, cfg_scale):
        self.model_full = model_full
        self.model_unlearned = model_unlearned
        self.w = w
        self.cfg_scale = cfg_scale

    def __call__(self, *args, **kwargs):
        """Forward pass with dynamic guidance"""
        # Override guidance in kwargs if needed
        kwargs['guidance_scale'] = self.cfg_scale
        
        # Get predictions from both models
        with torch.no_grad():
            # Run unlearned model
            output_unlearned = self.model_unlearned(*args, **kwargs, output_type='latent')
            
            # Run full model
            output_full = self.model_full(*args, **kwargs, output_type='latent')
            
            # Combine with dynamic weight
            latents = output_full.images + self.w * (output_unlearned.images - output_full.images)
            
            # Decode
            decoded = self.model_full.vae.decode(latents / self.model_full.vae.config.scaling_factor).sample
            decoded = (decoded + 1.0) / 2.0
            decoded = torch.clamp(decoded, 0.0, 1.0)
            
        return decoded


def generate_with_dynamic_w(
    prompt,
    model,
    model_orig,
    steps=28,
    guidance_scale=3.5,
    w=0,
    gen=None,
    device="cuda",
    height=512,
    width=512,
    max_sequence_length=512,
):
    """Generate image with dynamic guidance weight"""
    auto_model = AutoGuidedFluxModel(
        model_full=model_orig, model_unlearned=model, w=w, cfg_scale=guidance_scale
    )
    
    with torch.no_grad():
        img = auto_model(
            prompt=[prompt],
            prompt_2=[prompt],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
        )
    
    img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img_pil = to_pil_image((img_np * 255).astype(np.uint8))
    return img_pil


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    parser = argparse.ArgumentParser(
        description="Generate images with dynamic LoRA guidance weight for Flux"
    )
    parser.add_argument("--csv_path", type=str, default="I2P_prompts_4703.csv")
    parser.add_argument("--output_dir", type=str, default="generated_i2p")
    parser.add_argument(
        "--basemodel_id", type=str, default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument(
        "--alpha", type=float, default=8.0, help="LoRA alpha scaling factor"
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument(
        "--run_till_timestep", type=int, default=20, help="Timestep at which to compute latent diff"
    )
    parser.add_argument("--w1", type=float, default=-1.0, help="W1")
    parser.add_argument("--w2", type=float, default=2.0, help="W2")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("Start", flush=True)
    print(f"Using device: {args.device}")
    
    torch_dtype = torch.bfloat16
    
    if os.path.exists(os.path.join(args.output_dir, "train_config.json")):
        dirs = [args.output_dir]
    else:
        dirs = os.listdir(args.output_dir)

    for dirname in dirs:
        # Load prompts
        df = pd.read_csv(args.csv_path, index_col=0)
        exp_name = "_".join([str(item) for item in [
            "w1", args.w1, "w2", args.w2, "repeats", args.repeats, 
            "steps", args.num_inference_steps, "run_till", args.run_till_timestep
        ]])
        exp_dirpath = os.path.join(args.output_dir, dirname)
        os.makedirs(os.path.join(exp_dirpath, "images", exp_name), exist_ok=True)
        lora_path = os.path.join(exp_dirpath, "models", "lora.safetensors")
        
        if not os.path.exists(lora_path):
            print(f"Skip {dirname} - lora.safetensors not found")
            continue
        
        existing_images = len([f for f in os.listdir(os.path.join(exp_dirpath, "images", exp_name)) if f.endswith('.jpg')])
        if existing_images >= len(df):
            print(f"Skip {dirname} - already processed")
            continue
            
        print(f"Processing experiment: {dirname}.", flush=True)
        print(f"Existing images: {existing_images}", flush=True)

        # Load and prepare models
        model_orig, _ = load_flux_models(
            basemodel_id=args.basemodel_id, 
            torch_dtype=torch_dtype, 
            device=args.device
        )
        model, _ = load_flux_models(
            basemodel_id=args.basemodel_id,
            torch_dtype=torch_dtype,
            device=args.device
        )
        
        # Load LoRA weights
        from safetensors.torch import load_file
        lora_state_dict = load_file(lora_path)
        model.transformer.load_state_dict(lora_state_dict, strict=False)

        # Iterate over prompts
        for image_id, row in df.iterrows():
            image_path = os.path.join(exp_dirpath, "images", exp_name, f"{image_id:05d}.jpg")
            if os.path.exists(image_path):
                continue  # Skip if image already exists
            
            if image_id % WORLD_SIZE != RANK:
                continue
            
            prompt = row.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"Skip [{image_id}] empty prompt")
                continue
                
            start = time.time()
            seed = int(row.get("evaluation_seed", image_id))
            guidance = float(row.get("evaluation_guidance", args.guidance_scale))
            
            set_seed(seed)
            gen = torch.Generator(device=args.device).manual_seed(seed)

            prompt_diffs_arr = []
            empty_diffs_arr = []
            
            for repeat in range(args.repeats):
                prompt_diff = compute_latent_diff(
                    prompt,
                    model,
                    model_orig,
                    guidance,
                    seed,
                    repeats=1,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    run_till_timestep=args.run_till_timestep,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=args.max_sequence_length,
                    device=args.device,
                    torch_dtype=torch_dtype,
                )
                empty_diff = compute_latent_diff(
                    "",
                    model,
                    model_orig,
                    guidance,
                    seed,
                    repeats=1,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    run_till_timestep=args.run_till_timestep,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=args.max_sequence_length,
                    device=args.device,
                    torch_dtype=torch_dtype,
                )
                prompt_diffs_arr.extend(prompt_diff)
                empty_diffs_arr.extend(empty_diff)
                
            w = decide_w(prompt_diffs_arr, empty_diffs_arr, w1=args.w1, w2=args.w2)
            
            set_seed(seed)
            gen = torch.Generator(device=args.device).manual_seed(seed)
            
            img = generate_with_dynamic_w(
                prompt=prompt,
                model=model,
                model_orig=model_orig,
                steps=args.num_inference_steps,
                guidance_scale=guidance,
                w=w,
                gen=gen,
                device=args.device,
                height=args.image_size,
                width=args.image_size,
                max_sequence_length=args.max_sequence_length,
            )

            img.save(image_path)
            end = time.time()
            print(f"Prompt [{image_id}] (w={w:.2f}) processed in {end - start:.2f} seconds. Saved to {image_path}", flush=True)
