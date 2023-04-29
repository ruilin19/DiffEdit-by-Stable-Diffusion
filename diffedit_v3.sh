# horse demo
python get_edit_v3.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/white_horse.png \
    --avg_times 10 \
    --reference "a white horse" \
    --query "a zebra" \
    --output_dir ./v3_output \
    --strength 0.6 \
    --steps 25 \
    --scale 7.5 \
    --not_residual_guide \
# oragen demo
python get_edit_v3.py \
    --ckpt_dir  ./ckpt/runwayml_sd_v1_5 \
    --image_dir ./examples/oranges.png \
    --avg_times 10 \
    --reference "A basket of oranges" \
    --query "A basket of apples" \
    --output_dir ./v3_output \
    --strength 0.6 \
    --steps 25 \
    --scale 7.5 \
    --not_residual_guide \