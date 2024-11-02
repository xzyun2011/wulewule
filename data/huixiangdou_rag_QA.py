import asyncio
import logging
from datetime import datetime
import json
import re
import random
random.seed(0)
from tqdm import tqdm
import requests
from loguru import logger


from data_utils import response, save_json, save_json_once


def generate_answers(questions="黑神话悟空什么时候发布的",  english=False):
    """
        参考https://github.com/InternLM/HuixiangDou/blob/main/README_zh.md 先开启对应的服务
    """
    # 定义请求的URL
    url = 'http://127.0.0.1:23333/huixiangdou_inference'

    # 定义你要发送的数据
    data = {
        "text": f"{questions}",
        "image": ""
    }

    # 将数据转换成json格式
    payload = json.dumps(data)

    # 设置请求头
    headers = {
        'Content-Type': 'application/json'
    }

    # 发送POST请求
    rag_response = requests.post(url, data=payload, headers=headers)
    # 检查请求是否成功
    if rag_response.status_code != 200:
        logger.error(f"请求失败，状态码：{rag_response.status_code}")


    res_txt = rag_response.text
    res = json.loads(res_txt, strict=False)
    answer = ""
    for step in res["step"]:
        if step["state"] == "success":
            answer = step["response"]
            logger.info(f"got response: {answer}")
    if answer == "":
        logger.error(f'no answer: {res_txt}')
    return answer


def generate_rag_data(multi=False, simple_response=True):
    """
        multi: 是否生成多轮对话
        simple_response: 是否生成简略版回答，会修改prompt加入"。请简要回答。" 
    """
    aspects = [ "游戏概述", "章节与故事情节", "主要角色", "人物剧情梳理", "游戏世界观", "建筑与环境", "战斗系统", "游戏玩法", "艺术与音乐", "文化内涵", "市场影响", "彩蛋、网络梗" ] + [ "发售相关", "游戏背后的中国文化", "角色故事", "游戏攻略", "棍法类型", "天命人法术类型" ]
    # aspects = [ "发售相关", "人物和西游记关系", "角色广智", "取景地点", "陕北民谣", "游戏攻略", "棍法类型", "天命人法术类型" ]
    question_number = 10
    for ascpect in tqdm(aspects, total=len(aspects), desc="each aspects"):
        ## 一次多个问题，单轮对话
        if not multi:
            messages=[
                    {"role":"system", "content": f"""你是一名提问助手，专注于围绕游戏《黑神话：悟空》的特定方面（例如主要角色、战斗机制或故事情节）提出问题，20个字以内。
                                                根据用户的指示，调整你的提问内容，引导玩家深入思考该方面的背景、动机及其在游戏中的作用和发展。不需要序号，最终返回为一个list，格式为['问题', '问题', ...]"""},
                    {"role": "user", "content": f"请你就《黑神话：悟空》的‘{ascpect}’方面提出{question_number}个不同的问题，帮助玩家了解游戏在该方面的信息。"},
                    ]
            text_res = response(messages, temperature=0.7)
            try:
                questions = eval(text_res)
            except Exception as e:
                start_index = text_res.find('[')
                end_index = text_res.find(']')
                text_res = text_res[start_index:end_index+1]
                questions = eval(text_res)
            except Exception as e:
                logger.error(f"Got exception {e}, text_res:\n{text_res}")
                raise ValueError("text res must be list")
            logger.info(f"questions: {questions}")

            # 生成回答
            for question in tqdm(questions, total=len(questions), desc="generating answer"):
                if simple_response:
                    question += "。请简要回答。" 
                answer = generate_answers(questions=question, english=False)

                # 保存结果
                if len(answer) > 10:
                    conversation = {"conversation": 
                            [
                                {
                                    "system": base_system_propmt, ##not used
                                    "input": question,
                                    "output": answer, 
                                }
                            ]
                        }

                    save_json_once(conversation, save_path)
        ## 某个方面，多轮对话
        else:
            question_number = 1
            iters = random.randint(2, 4)
            conversation = {"conversation":[] }
            question, answer = "", ""
            for i in range(iters):                 
                messages=[
                        {"role":"system", "content": f"""你现在是一个专门的提问助手，目标是帮助玩家深入探索《黑神话：悟空》这款游戏。你的任务是基于玩家的回答，提出越来越细致和深入的问题。你的提问风格应当简洁、清晰，20个字以内。"""},
                        {"role": "user", "content": f"请你就《黑神话：悟空》的‘{ascpect}’方面提出{question_number}个问题，帮助玩家了解游戏在该方面的信息。"},
                        ]
                if i > 0:
                    messages += [
                                {"role": "assistant", "content": f"{question}"},
                                {"role": "user", "content": f"{answer}"},
                                ]
                question = response(messages, temperature=0.7)
                if not ("黑神话" in question):
                    question = "在《黑神话：悟空》中，" + question
                logger.info(f"multi turn, questions:\n{question}")
                res =[""]
                if simple_response:
                    question += "。请简要回答。" 
                answer = generate_answers(questions=question, english=False)
                logger.info(f"multi turn, answer:\n{answer}")
                if len(conversation["conversation"]) == 0 and len(answer) > 10:
                    conversation["conversation"].append({
                                "system": base_system_propmt, ##not used
                                "input": question,
                                "output": answer, 
                            })
                elif ( len(conversation["conversation"]) > 0 and len(answer) > 10):
                    conversation["conversation"].append({
                                "input": question,
                                "output": answer, 
                            })
            if len(conversation["conversation"]) > 0:
                save_json_once(conversation, save_path)




if __name__ == '__main__':

    save_path = "./huixiangdou_conversations.jsonl"
    base_system_propmt = "你是悟了悟了，由xzyun2011开发的AI助手，专注于回答和《黑神话：悟空》这款游戏相关的问题，你想帮助玩家了解更多这款游戏背后的故事和文化知识。"
    # # base_system_propmt = """You are Wulewule, an AI assistant developed by xzyun2011. Your primary focus is to answer questions related to the game "Black Myth: Wukong".  You aim to assist players in learning more about the game's storyline, cultural significance, and background. """
    for i in range(1):
        random.seed(i)
        multi = ( random.random()>0.5 )
        generate_rag_data(multi = multi, simple_response=True)
        # generate_rag_data(multi=True)
