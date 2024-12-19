# 模型微调与量化

## 1 硬件要求
由于模型微调过程和autoawq量化过程只能在gpu上进行计算，因此需要nvidia或amd的gpu,这里我们使用的是nvidia A6000显卡来进行模型的微调和量化,CUDA版本为12.0。

## 2 目录结构
```
ft_quant
|—— README.md                   # 本文档
|—— LLaMA-Factory               # 微调工具(待生成)
|—— requirements.txt            # 量化环境依赖
|—— quant.py                    # 量化脚本
|—— MODELS                      # 模型
    |—— Qwen-1_8B-sft-adapter       # 微调权重适配器(待生成)
    |—— Qwen-1_8B-finetuned         # 完成权重合并的微调后模型(待生成)
    |—— Qwen-1_8B-quantized         # 量化后模型(待生成)
|—— llama.cpp                   # 格式转换与模型量化工具(待生成)
|—— 
```

## 3 模型微调

### 3.1 环境配置
```
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n ft_quant python=3.10
conda activate ft_quant
cd LLaMA-Factory
pip install -e .[metrics]
```

### 3.2 模型微调与权重合并
微调指令：
```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-1_8B \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir ../MODELS/Qwen-1_8B-sft-adapter \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

权重合并指令：
```
CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path Qwen/Qwen-1_8B \
    --adapter_name_or_path ../MODELS/Qwen-1_8B-sft-adapter \
    --template default \
    --finetuning_type lora \
    --export_dir ../MODELS/Qwen-1_8B-finetuned \
    --export_size 2 \
    --export_legacy_format False
```

完成微调和权重合并的新模型保存在`ft_quant/MODELS/Qwen-1_8B-finetuned`，模型结构与Qwen-1_8B完全相同，可在此基础上进行后续的量化步骤。

## 4 模型量化

### 4.1 量化过程
执行以下命令:
```sh
cd ..   ## 回到ft_quant目录下
pip3 install -r requirements.txt
python3 quant.py #需要python3.9以上

#此处和原始llama.cpp有些区别，原始llama.cpp的Q4_0量化的output.weight是Q6_k类型，我们将其调整为了Q4_0类型，所以这一步请使用我们修改好的量化
cd AICAS/llama-cpp-python/vender/llama.cpp && make -j8
./quantize AICAS/ft_quant/MODELS/Qwen-1_8B-finetuned/ggml-model-f16.gguf
#生成模型ggml-model-Q4_0.gguf