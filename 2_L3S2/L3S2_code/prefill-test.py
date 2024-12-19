import pexpect
import re
import time
from settings import *

def extract_number(text):
    # 定义正则表达式来匹配数字
    pattern = r"\d+"
    # 在字符串中搜索匹配的数字
    match = re.search(pattern, text)
    # 如果找到匹配，返回提取的数字，否则返回None
    if match:
        return int(match.group())  # 将匹配的字符串转换为整数并返回
    else:
        return None  # 没有找到数字时返回None

cmd_prefill = f"{main_path} -m {gguf_path} \
        -f prompt.txt \
        -c 620 -n 1 --ignore-eos -t 8 -tb 8"

child = pexpect.spawn(cmd_prefill)
# 排除掉prefill之前的输出部分,恰好有9个'>'
for _ in range(9):
    child.expect('>', timeout=None)
child.expect("<", timeout=None)
# 记录开始的时间
start_time = time.time()
child.expect("l", timeout=None)
# 记录结束的时间
end_time = time.time()
child.expect("T", timeout=None)
child.expect(">", timeout=None)
output_before_match = child.before.decode('utf-8') 
prefill_tokens = extract_number(output_before_match)
prefill_speed = prefill_tokens / (end_time - start_time)

print(f"total prefill tokens: {prefill_tokens}")
print(f"total time to prefill: {end_time - start_time} seconds")  
print(f"prefill throught: {prefill_speed:.2f} tokens / seconds")  
