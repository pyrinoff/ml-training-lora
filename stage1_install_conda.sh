conda create -n alpaca-lora python=3.10 -y
conda activate alpaca-lora
pip install -r requirements.txt
cp /opt/conda/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so /opt/conda/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so
