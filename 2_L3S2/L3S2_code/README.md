## 团队信息 （L3S2队）
|团队成员|参赛ID|学校|
|---|---|---|
|刘杰（指导老师）|TinyJie|南京大学|
|梁锦文|R桑从不撒谎|南京大学|
|罗星宇|Confetti|南京大学|
|石璐|Polluxy|南京大学|
|孙源泽|Luca_Suen|南京大学|

## 模型优化
由于autoawq量化过程只能在gpu上进行计算，因此需要nvidia或amd的gpu,这里我们使用的是nvidia A6000显卡来进行的量化,CUDA版本为12.0。

详情见./ft_quant/README.md


## 精度测试
```sh
cd AICAS/lm-evaluation-harness
conda activate AICAS

"""
准确度测试是使用了llama-cpp-python库来执行llama.cpp的推理过程，我们修改了一部分llama-cpp-python库的数据后处理过程（排序的优化），将准确度测试时间减少到了一个小时左右，可以在短时间内完成相应测试。
"""
export HF_ENDPOINT=https://hf-mirror.com

lm_eval\
--model my_gguf\
--model_args model=/root/data/loramerged-quantized-q40/ggml-model-Q4_0.gguf\
--tasks piqa\
--output_path ./lm-eval-output/output_precision
```
最终结果保存在/root/AICAS/lm-evaluation-harness/lm-eval-output/output_precision

## 吞吐率和内存占用测试
```sh
cd AICAS
conda activate AICAS


# 杀除一些monitor无法杀尽的进程，如果一次没杀干净可以多杀几次
killall -9 python
killall -9 cpptools

"""
吞吐率的测试是直接调用的llama.cpp编译生成的main程序，使用pexpect开源库执行C程序和定位输出（我们在AICAS/llama-cpp-python/vender/llama.cpp/example/main/main.cpp添加了两处打印内容，结合其原本的输出来定位prefill和decode的开始结束时机）

内存方面检测的是运行过程中的./main进程

benchmark.py先进行decode吞吐率的测试然后再进行prefill吞吐率的测试（设定--ignore-eos、-n和-ins运行项的区别），monitor只检测decode吞吐率测试过程，因为在decode吞吐率测试过程中包含了prefill过程.

可以使用python monitor.py prefill-test.py来单独测试prefill的内存和吞吐率

setting.py用于确定测试的代码文件路径和模型文件路径

详细信息可见代码
"""
python monitor.py
python benchmark.py
```
吞吐率最终结果保存在/root/AICAS/throughput_results.json

内存占用最终结果保存在/root/AICAS/memory_results.json

