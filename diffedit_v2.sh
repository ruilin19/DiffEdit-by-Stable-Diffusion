# horse demo
python get_mask_v2.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/white_horse.png \
    --avg_times 10 \
    --reference "a white horse" \
    --query "a zebra" \
    --output_dir ./v2_output \
    --strength 0.6 \
    --steps 25 \
    --scale 7.5 \
    --not_residual_guide \

python inpaint.py \
    --ckpt_dir  ./ckpt/inpainting-v1-2 \
    --image_dir ./examples/white_horse.png \
    --mask_dir ./v2_output/mask_v2.png \
    --query "a zebra" \
    --output_dir ./v2_output \
    --steps 25 \
    --scale 7.5 \

# orange demo
python get_mask_v2.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/oranges.png \
    --avg_times 10 \
    --reference "A basket of oranges" \
    --query "A basket of apples" \
    --output_dir ./v2_output \
    --strength 0.6 \
    --steps 25 \
    --scale 7.5 \
    --not_residual_guide \

python inpaint.py \
    --ckpt_dir  ./ckpt/inpainting-v1-2 \
    --image_dir ./examples/oranges.png \
    --mask_dir ./v2_output/mask_v2.png \
    --query "A basket of apples" \
    --output_dir ./v2_output \
    --steps 25 \
    --scale 7.5 \