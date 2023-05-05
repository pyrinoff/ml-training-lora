apt-get install zip
zip -r output-lora-alpaca.zip output-lora-alpaca/
echo "OUT CAN DOWNLOAD YOUR WEIGHTS NOW!!!"
echo "GENERATING TEST OUTPUT:"
python generate.py \
  --base_model='yahma/llama-7b-hf' \
  --lora_weights="./output-lora-alpaca/" \
  --load_8bit \
  --input_file="./generate/generate_tasks.json"