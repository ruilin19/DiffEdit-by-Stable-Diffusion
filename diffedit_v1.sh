# horse demo
python get_mask_v1.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/white_horse.png \
    --avg_times 10 \
    --reference "a white horse" \
    --query "a zebra" \
    --output_dir ./v1_output \
    --strength 0.8 \
    --steps 50 \
    --scale 7.5 \

python inpaint.py \
    --ckpt_dir  ./ckpt/inpainting-v1-2 \
    --image_dir ./examples/white_horse.png \
    --mask_dir ./v1_output/mask_v1.png \
    --query "a zebra" \
    --output_dir ./v1_output \
    --steps 50 \
    --scale 7.5 \
# orange demo
python get_mask_v1.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/oranges.png \
    --avg_times 10 \
    --reference "A basket of oranges" \
    --query "A basket of apples" \
    --output_dir ./v1_output \
    --strength 0.8 \
    --steps 50 \
    --scale 7.5 \

python inpaint.py \
    --ckpt_dir  ./ckpt/inpainting-v1-2 \
    --image_dir ./examples/oranges.png \
    --mask_dir ./v1_output/mask_v1.png \
    --query "A basket of apples" \
    --output_dir ./v1_output \
    --steps 50 \
    --scale 7.5 \