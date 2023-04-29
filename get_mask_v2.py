import os
import torch
import argparse
from PIL import Image
from skimage import morphology
from diffusers import DDIMScheduler
from modules.diffedit_v2 import DiffEdit_v2

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
        '--avg_times',
        default=10,
        type=int,
        help='times of caculate the difference to attain mask'
    )
    parser.add_argument(
        '--reference',
        default=None,
        type=str,
        help='reference prompt'
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
        '--strength',
        default=0.6,
        type=float,
        help='hyperparamemter of pipeline'
    )
    parser.add_argument(
        '--steps',
        default=25,
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
    parser.add_argument(
        '--not_residual_guide',
        default=False,
        action="store_true",
        help="whether to use the noise residual to compute the mask or not"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    DDIM =  DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        subfolder="scheduler"
    )
    pipe = DiffEdit_v2.from_pretrained(
        pretrained_model_name_or_path=args.ckpt_dir,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=DDIM,
    ).to("cuda")

    # save memory and inference fast
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    image = Image.open(args.image_dir).convert('RGB').resize((512, 512))

    pil_mask = pipe.get_mask(
        latents_num=args.avg_times,
        refer_prompt=args.reference,
        query_prompt=args.query,
        image=image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        seed=args.seed,
        residual_guide=not args.not_residual_guide
    )
    uncleaned_mask = Image.fromarray(pil_mask).convert('RGB')
    pil_mask = morphology.remove_small_objects(pil_mask, min_size=16)
    cleaned_mask = Image.fromarray(pil_mask).convert('RGB')
    resized_cleaned_mask = Image.fromarray(pil_mask).convert('RGB').resize((512, 512))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    uncleaned_mask.save(f'{args.output_dir}/unclean_mask_v2.png')
    cleaned_mask.save(f'{args.output_dir}/clean_mask_v2.png')
    resized_cleaned_mask.save(f'{args.output_dir}/mask_v2.png')