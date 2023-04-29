import os
import torch
import argparse
from PIL import Image
from diffusers import DDIMScheduler,StableDiffusionInpaintPipeline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default=None,
        type=str,
        help='path of the weight of stable diffusioninpaint pipeline '
    )
    parser.add_argument(
        '--image_dir',
        default=None,
        type=str,
        help='path of the image to be edited'
    )
    parser.add_argument(
        '--mask_dir',
        default=None,
        type=str,
        help='path of the mask'
    )
    parser.add_argument(
        '--query',
        default=None,
        type=str,
        help='edit prompt'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        help='path of the result to be saved'
    )
    parser.add_argument(
        '--steps',
        default=50,
        type=int,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--seed',
        default=2625,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--scale',
        default=7.5,
        type=float,
        help='hyperparamemter of pipeline'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    DDIM =  DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        subfolder="scheduler"
    )
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=DDIM,
    ).to("cuda")

    # save memory and inference fast
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    # pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()

    image = Image.open(args.image_dir).convert('RGB').resize((512, 512))
    mask = Image.open(args.mask_dir).convert('RGB').resize((512, 512))

    result = pipe(
        prompt=args.query,
        image=image,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed)
    ).images[0]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    result.save(f'{args.output_dir}/result.png')