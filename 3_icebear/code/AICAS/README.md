测评步骤
1. 到 build 目录编译
   
   cmake .. -DCMAKE_C_FLAGS=-march=armv8.6-a && cmake --build . --config Release
   
2. 起服务

 ./bin/aicas -m /mnt/disk/Qwen-1_8B-Short5/ggml-model-q8_0.gguf -t 8 -b 560 -c 600 -ctk q8_0 -ub 64

3. 监听进程内存状态

  python monitor.py aicas

4. 测试

  python benchmark.py
