# LLaMA2-70B LoRA finetuning

0. Set MODEL env var to the LLaMa2-70B path in HF format.
```
export MODEL=<path>
```

1. Build and run container
```
./docker_build.sh && ./docker_run.sh
```

2. Prepare dataset
```
python3 prepare_dataset.py
```

3. Convert model into nemo format
```
python NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=/model/llama2-70b-hf/ --out-file=./model/llama2-70b.nemo
```

4. Run training
```
./run.sh
```