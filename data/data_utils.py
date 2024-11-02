from openai import OpenAI 
import time
import os
import json

# siliconflow
siliconflow_interlm = {
    "api_key": "your-api",
    "base_url": "https://api.siliconflow.cn/v1",
    # "model_name": "Qwen/Qwen2-Math-72B-Instruct",
    # "model_name": "internlm/internlm2_5-20b-chat",
    "model_name": "deepseek-ai/DeepSeek-V2.5",
    "temperature": 0.5,
}

client = OpenAI(
        api_key=siliconflow_interlm["api_key"],
        base_url=siliconflow_interlm["base_url"],
    )

def response(messages, max_tokens=512, temperature=0.5, sleep_time=0.05):
    try:
        res = client.chat.completions.create(
            model=siliconflow_interlm["model_name"],
            messages=messages,
            # max_tokens=max_tokens,
            temperature=temperature,
            ).choices[0].message.content
    except Exception as e:
        print(f"Got exception {e}, sleep(0.05s) and retry again")
        if sleep_time>0:
            time.sleep(sleep_time)
        res = client.chat.completions.create(
            model=siliconflow_interlm["model_name"],
            messages=messages,
            # max_tokens=max_tokens,
            temperature=temperature,
            ).choices[0].message.content
    return res


def save_json(data_list, file_path, jsonl=False):
    dirname, filename = os.path.split(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if filename.split(".")[-1] == "jsonl":
        jsonl = True
    with open(file_path, 'at', encoding='utf-8') as f:
        if jsonl and isinstance(data_list, list):
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False, indent=4) + '\n')
        else:
             f.write(json.dumps(data_list, ensure_ascii=False, indent=4))


def save_json_once(data, file_path):
    dirname, filename = os.path.split(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(file_path, 'at', encoding='utf-8') as f:
        ## 不加 indent，单条数据就是1行
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

# file_names = [ "游戏概述", "章节与故事情节", "主要角色", "人物剧情梳理", "游戏世界观", "建筑与环境", "战斗系统", "游戏玩法", "艺术与音乐", "文化内涵", "市场影响", "彩蛋梗数据" ]