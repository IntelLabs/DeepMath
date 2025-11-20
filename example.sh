#########################################
# Training
#########################################

# VLLM server
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    python -m deep_math.vllm_serve_agent \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --tensor_parallel_size 1 \
    --data_parallel_size 6 \
    --config configs/vllm_agent_server_4B.yaml \
    --max_steps 50 \
    --max_agent_output 5000

# Trainer
CUDA_VISIBLE_DEVICES=6,7 \
    accelerate launch --config_file configs/zero2.yaml \
    --num_processes 2 \
    training.py


#########################################
# Inference
#########################################
# All code examples refer to the MATH500 dataset.

# Baseline
python inference.py -m hydra/launcher=slurm \
        +model.use_vllm=true \
        model.generation.max_new_tokens=20000 model.generation.do_sample=true \
        model.generation.temperature=0.6 model.generation.top_p=0.95 model.generation.top_k=20 model.sampling=16 \
        model.template=deep_seek_math \
        model.model_name_or_path=Qwen/Qwen3-4B-Thinking-2507 \
        hf_tag=HuggingFaceH4/MATH-500 \
        generated_file=baseline-20k-MATH500-vllm-qwen3-4B.jsonl

# LORA
python inference.py -m hydra/launcher=slurm \
       +model.use_vllm=true \
       model.generation.max_new_tokens=20000 model.generation.do_sample=true \
       model.generation.temperature=0.6 model.generation.top_p=0.95 model.sampling=16 model.generation.top_k=20 \
       model.template=deep_seek_math \
       model.model_name_or_path=Qwen/Qwen3-4B-Thinking-2507 \
       model.lora_path=/PATH/TO/CHECKPOINT \
       +model.vllm_params.max_lora_rank=32 \
       hf_tag=HuggingFaceH4/MATH-500 \
       generated_file=trained-model-no-agent-20k-MATH500-Qwen3-4B.jsonl

# Agent use
python inference.py -m hydra/launcher=slurm \
       +model.use_vllm=true \
       model.generation.max_new_tokens=3000 +model.max_steps=50  +model.max_agent_output=20000 model.generation.do_sample=true \
       model.generation.temperature=0.6 model.generation.top_p=0.95 model.generation.top_k=20 model.sampling=16 \
       model.template=simple_template \
       model.instruction=agent_math_instruction_2 \
       +model.math_agent=true \
       +model.examples=deep_math/fewshot.txt \
       model.model_name_or_path=Qwen/Qwen3-4B-Thinking-2507 \
       hf_tag=HuggingFaceH4/MATH-500 \
       generated_file=agent-20k-MATH500-qwen3-4B.jsonl

# Agent w/ LORA
python inference.py -m hydra/launcher=slurm \
       +model.use_vllm=true \
       +model.vllm_params.max_lora_rank=32 \
       model.generation.max_new_tokens=3000 +model.max_steps=50 +model.max_agent_output=20000 model.generation.do_sample=true \
       model.generation.temperature=0.6 model.generation.top_p=0.95 model.sampling=16 model.generation.top_k=20 \
       model.template=simple_template \
       model.instruction=agent_math_instruction_2 \
       +model.math_agent=true \
       +model.examples=deep_math/fewshot.txt \
       model.model_name_or_path=Qwen/Qwen3-4B-Thinking-2507 \
       model.lora_path=/PATH/TO/CHECKPOINT \
       hf_tag=HuggingFaceH4/MATH-500 \
       generated_file=trained-model-with-agent-20k-MATH500-Qwen3-4B.jsonl

#########################################
# Evaluation
#########################################
python evaluation.py -m generated_file=baseline-20k-MATH500-vllm-qwen3-4B.jsonl,trained-model-no-agent-20k-MATH500-Qwen3-4B.jsonl,agent-20k-MATH500-qwen3-4B.jsonl,trained-model-with-agent-20k-MATH500-Qwen3-4B.jsonl