from openai import OpenAI 
import time
import random
random.seed(0)
from tqdm import tqdm
import os
import json
import argparse

from data_utils import response, save_json, save_json_once

def generate_questions(english=False):
    questions = []
    question_types =["你是谁呀？", "你好呀，你有啥功能？"]
    if english:
        question_types =["Who are you?", "Hello there, what can you do?"]
    question_number = 15
    print(f"start generating questions...")
    for question_type in tqdm(question_types):
        # question_prompt = f"你是一个ai助手，用户刚开始和你对话会说什么？请你简单模拟，只列出用户{question_type}。列举{question_number}个类似{question_type}，仅返回一个列表，输出格式为['提问', '提问', ...]。"
        question_prompt = f"请帮我把'{question_type}'换不同方式讲，不改变其本意，只返回{question_number}个类似的句子，输出格式为['句子', '句子', ...]。"
        if english:
            question_prompt = f"Please help me rephrase '{question_type}' in different ways without changing its meaning. Return {question_number} similar sentences, and format the output as ['sentence', 'sentence', ...]."
        messages =[ {"role":"system", "content": question_prompt},]
        text_res = response(
            messages=messages,
            temperature=0.7)
        try:
            list_res = eval(text_res)
        except Exception as e:
            start_index = text_res.find('[')
            end_index = text_res.find(']')
            text_res = text_res[start_index:end_index+1]
            list_res = eval(text_res)
        except Exception as e:
            print(f"Got exception {e}, text_res:\n{text_res}")
            raise ValueError("text res must be list")
        print(f"type:{question_type} \n {str(list_res)}")
        questions += list_res
    return questions

def generate_answers(questions, base_system_propmt, english=False):
    used_number = min(15, len(questions))
    questions_sampled = random.sample(questions, used_number)
    print(f"Start generating answers...")
    answers = []
    answer_prompt = f"{base_system_propmt}请牢记你的这些设定。无论用户问你什么，你只按设定简单介绍自己，可以重新组织语言来介绍。准备好了回复明白。"
    first_answer = f"明白。{base_system_propmt.replace('你','我')}"
    if english:
        answer_prompt = f"""{base_system_propmt}Remember your setting at all times. No matter what the user inquires about, 
        simply introduce yourself based on these settings, and feel free to rephrase your introduction. Reply with 'Understood' when prepared."""
        first_answer = f"Understood. {base_system_propmt.replace('You are','I am').replace('Your','My').replace('You','I').replace('your','my').replace('you','I')}"
    for question in tqdm(questions_sampled):
        text_res = response(messages =[
            {"role":"system", "content": answer_prompt},
            {"role": "assistant", "content": first_answer},
            {"role": "user", "content": question},
            ],            
            temperature=0.5)
        answers.append(text_res.replace("\n", ""))
    print(f"answers:{answers} \n {answers}")
    return answers


def generate_selfcognition_data(save_path="./self_cognition.jsonl", ai_name="悟了悟了", author="xzyun2011", english=False):
    base_system_propmt = f"你是{ai_name}，由{author}开发的AI助手，专注于回答和《黑神话：悟空》这款游戏相关的问题，你想帮助玩家了解更多这款游戏背后的故事和文化知识。"
    if english:
        base_system_propmt = f"""You are Wulewule, an AI assistant developed by {author}. Your primary focus is to answer questions related to the game 'Black Myth: Wukong'. You aim to assist players in learning more about the game's storyline, cultural significance, and background."""
    questions = generate_questions(english)
    answers = generate_answers(questions, base_system_propmt, english)
    print(f"Start generating conversations...")
    ## prepare conversations
    conversations = [ ]
    for question, answer in zip(questions, answers):
    # for question in questions:
    #     for answer in answers:
        conversation_i = {"conversation": 
            [
                {
                    "system": base_system_propmt, ##not used
                    "input": question,
                    "output": answer, 
                }
            ]
        }
        conversations.append(conversation_i)
        save_json_once(conversation_i, save_path)
    # save_json(conversations, save_path)
    print(f"Done, conversations saved in {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate self cognition dataset')
    parser.add_argument('--save-path', type=str, default="./self_cognition.jsonl", help='json file save path')
    parser.add_argument('--ai-name', type=str, default="悟了悟了", help='ai name for system prompt')
    parser.add_argument('--author', type=str, default="xzyun2011", help='author name for system prompt')
    parser.add_argument("--en", "--English", "--english", action="store_true", help="generate English self cognition data")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.en:
        print("================== Generating English dataset ==================")
    generate_selfcognition_data(args.save_path, args.ai_name, args.author, args.en)

if __name__ == '__main__':
    main()
