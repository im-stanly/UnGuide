import os
import json
import argparse
from functools import partial
import random
import copy

import numpy as np
import torch

from utils import load_flux_models, print_trainable_parameters, set_seed
from tqdm import tqdm

from ldm.util import instantiate_from_config
from sampling import sample_model
from lora import LoRALinear, inject_lora_nsfw, inject_lora

import sys
from diffusers import FluxPipeline

sys.path.append('.')
from utils.flux_utils import esd_flux_call
FluxPipeline.__call__ = esd_flux_call

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Flux Model"
    )

    # Model configuration
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/stable-diffusion/v1-inference.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./models/sd-v1-4.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--basemodel_id", default="black-forest-labs/FLUX.1-dev", help="Base model ID for Flux"
    )

    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument(
        "--lora_alpha", type=int, default=8, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["attn2.to_k", "attn2.to_v"],
        help="Target modules for LoRA injection",
    )

    # Training configuration
    parser.add_argument(
        "--iterations", type=int, default=200, help="Number of training iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--image_size", type=int, default=512, help="Image size for training"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="DDIM sampling steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
        "--num_inference_steps", type=int, default=28, help="Number of inference steps"
        "--max_sequence_length", type=int, default=512, help="Max sequence length for embeddings"
        "--batchsize", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0, help="DDIM eta parameter"
    )
    parser.add_argument(
        "--start_guidance", type=float, default=9.0, help="Starting guidance scale"
        "--guidance_scale", type=float, default=1.0, help="Training guidance scale"
    )
    parser.add_argument(
        "--negative_guidance", type=float, default=1.0, help="Negative guidance scale"
    )
    parser.add_argument(
        "--inference_guidance_scale", type=float, default=3.5, help="Inference guidance scale"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="data/cat.json",
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--save_losses", action="store_true", help="Save training losses to file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== LoRA Fine-tuning Configuration ===")
    print(f"Base Model: {args.basemodel_id}")
    print(f"Device: {args.device}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Training iterations: {args.iterations}")
    print(f"Learning rate: {args.lr}")
    print("=" * 40)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load prompts json
    with open(args.prompts_json, 'r') as f:
        prompts_data = json.load(f)

    target_prompt = prompts_data.get("target", None)
    reference_prompt = prompts_data.get("reference", None)
    if target_prompt is None or reference_prompt is None:
        raise ValueError(f"Missing required prompt")
    
    config = {  
        "basemodel_id": args.basemodel_id,
        "lora_rank": args.lora_rank,
        "iterations": args.iterations,
        "lr": args.lr,
        "image_size": args.image_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "inference_guidance_scale": args.inference_guidance_scale,
        "negative_guidance": args.negative_guidance,
        "target_prompt": target_prompt,
        "reference_prompt": reference_prompt,
        "prompts_json": args.prompts_json,
        "class_name": args.prompts_json.split("/")[-1].split(".")[0],
    }
    
    dir_name = "_".join(f"{k}_{config[k]}" for k in [
        "class_name",
        "lora_rank",
        "iterations",
        "lr",
        "guidance_scale",
        "negative_guidance",
        "num_inference_steps"
    ])
    os.makedirs(os.path.join(args.output_dir, dir_name, "models"), exist_ok=True)

    # Initialize models
    print("Loading models...")
    torch_dtype = torch.bfloat16
    model, model_orig = load_flux_models(
        basemodel_id=args.basemodel_id,
        torch_dtype=torch_dtype,
        device=args.device,
        lora_rank=args.lora_rank
    )

    model.set_progress_bar_config(disable=True)

    # Freeze original model parameters
    for pipeline in [model, model_orig]:
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)

    # Get trainable parameters (only LoRA layers)
    print_trainable_parameters(model)

    # Initialize training components
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.transformer.parameters()),
        lr=args.lr
    )
    criterion = torch.nn.MSELoss()
    losses = []

    height = width = args.image_size
    batchsize = args.batchsize

    # Prepare prompt embeddings
    print("Encoding prompts...")
    prompts = [target_prompt, reference_prompt, reference_prompt]
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all, text_ids = model.encode_prompt(
            prompts, prompt_2=prompts, max_sequence_length=args.max_sequence_length
        )

        target_prompt_embeds, reference_prompt_embeds, reference_prompt_embeds_neg = prompt_embeds_all.chunk(3)
        target_pooled_embeds, reference_pooled_embeds, reference_pooled_embeds_neg = pooled_prompt_embeds_all.chunk(3)

        # Prepare random latent input
        model_input = model.vae.encode(
            torch.randn((1, 3, height, width)).to(torch_dtype).to(model.vae.device)
        ).latents.cpu()

    # Move text encoders and VAE to CPU to save memory
    model.text_encoder_2.to('cpu')
    model.text_encoder.to('cpu')
    model.vae.to('cpu')

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Training loop
    print("Starting training...")
    pbar = tqdm(range(args.iterations))

    for i in pbar:
        optimizer.zero_grad()

        guidance = torch.tensor([args.guidance_scale], device=args.device)
        guidance = guidance.expand(batchsize)

        # Sample random timestep
        run_till_timestep = random.randint(0, args.num_inference_steps - 1)
        timesteps = model.scheduler.timesteps[run_till_timestep].unsqueeze(0).to(args.device)
        seed = random.randint(0, 2**15)

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            args.device,
            torch_dtype,
        )

        model.transformer.eval()
        with torch.no_grad():
            # Generate intermediate latent
            xt = model(
                prompt_embeds=reference_prompt_embeds,
                pooled_prompt_embeds=reference_pooled_embeds,
                num_images_per_prompt=batchsize,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.inference_guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=torch.Generator().manual_seed(seed),
                output_type='latent',
                height=height,
                width=width,
            ).images

            # Get predictions from original model
            noise_pred_reference_neg = model_orig.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=reference_pooled_embeds_neg,
                encoder_hidden_states=reference_prompt_embeds_neg,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            noise_pred_reference = model_orig.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=reference_pooled_embeds,
                encoder_hidden_states=reference_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            noise_pred_target = model_orig.transformer(
                hidden_states=xt,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_pooled_embeds,
                encoder_hidden_states=target_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        model.transformer.train()
        # Get prediction from trainable model
        model_pred = model.transformer(
            hidden_states=xt,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=reference_pooled_embeds,
            encoder_hidden_states=reference_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # Compute loss (negative guidance objective)
        target = noise_pred_reference - args.negative_guidance * (noise_pred_target - noise_pred_reference_neg)
        loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record loss
        loss_value = loss.item()
        losses.append(loss_value)
        pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        # Cleanup
        model_pred = loss = target = xt = None
        torch.cuda.empty_cache()
        gc.collect()

    # Save trained model
    print(f"Saving trained model to {args.output_dir}/{dir_name}/models")
    lora_path = os.path.join(args.output_dir, dir_name, "models", "lora.safetensors")
    model.transformer.save_pretrained(lora_path)

    print("Training completed!")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss: {sum(losses) / len(losses):.6f}")
    
    config["final_loss"] = losses[-1]
    config["average_loss"] = sum(losses) / len(losses)
    
    with open(os.path.join(args.output_dir, dir_name, "train_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {os.path.join(args.output_dir, dir_name, 'train_config.json')}")


if __name__ == "__main__":
    main()

