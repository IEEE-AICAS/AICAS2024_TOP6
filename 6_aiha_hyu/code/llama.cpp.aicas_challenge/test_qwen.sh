# F16 Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16.gguf        -t 1 -tb 1 --n-predict 20 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# F16 Test (8 Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0

# Q8_0 Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 128 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q8_0 Test (8 Thread)
# for i in {1..5}
# do
#    ./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0.gguf -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0
# done

# Q8_0_H Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_h.gguf        -t 1 -tb 1 --n-predict 20 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q8_0_H Test (8 Thread)
# for i in {1..5}
# do
#      ./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_h.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0
# done

# Q8_0_S Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_s.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q8_0_S Test (8 Thread)
# for i in {1..5}
# do
#     ./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_s.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0
# done

# Q8_0_F Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_f.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q8_0_F Test (8 Thread)
# for i in {1..5}
# do
#     ./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q8_0_f.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0
# done

# Q4_0 Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q4_0 Test (8 Thread)   
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0

# Q4_0_H Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_h.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q4_0_H Test (8 Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_h.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0

# Q4_0_S Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_s.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q4_0_S Test (8 Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_s.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q4_0_F Test (Single Thread)
#./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_f.gguf        -t 1 -tb 1 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt_short_one.txt" --ignore-eos --temp 0

# Q4_0_F Test (8 Thread)
./main -m /root/workspace_hmk/llama.cpp/Qwen-1_8B/ggml-model-f16_q4_0_f.gguf        -t 8 -tb 8 --n-predict 100 --ctx_size 1024 --file "/root/workspace_hmk/llama-cpp-python/prompt.txt" --ignore-eos --temp 0
