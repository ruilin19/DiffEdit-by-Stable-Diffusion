mkdir runwayml_sd_v1_5
cd runwayml_sd_v1_5
mkdir feature_extractor
wget -P ./feature_extractor https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json

mkdir scheduler
wget -P ./scheduler https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/scheduler_config.json

mkdir text_encoder
wget -P ./text_encoder https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json
wget -P ./text_encoder https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin

mkdir tokenizer
wget -P ./tokenizer https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt
wget -P ./tokenizer https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json
wget -P ./tokenizer https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json
wget -P ./tokenizer https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json

mkdir unet
wget -P ./unet https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json
wget -P ./unet https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin

mkdir vae
wget -P ./vae https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/config.json
wget -P ./vae https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin

wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json