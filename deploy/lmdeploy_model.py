from lmdeploy.model import MODELS, BaseChatTemplate

@MODELS.register_module(name='customized_model')
class CustomizedModel(BaseChatTemplate):
    """A customized chat template."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='You are a robot developed by LMDeploy.',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|action_end|>']):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words)
        

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import streamlit as st
from lmdeploy import TurbomindEngineConfig, pipeline, GenerationConfig, ChatTemplateConfig
from lmdeploy.serve.async_engine import AsyncEngine
from modelscope import snapshot_download
import logging
from typing import Any, List, Optional, Iterator
import hydra

from download_models import download_model
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TextIteratorStreamer

class LmdeployLM(LLM):   
    llm_system_prompt: str=""
    model: AsyncEngine = None
    gen_config: GenerationConfig = None
    def __init__(self, model_path: str, llm_system_prompt: str, cache_max_entry_count: float):
        super().__init__()
        self.llm_system_prompt = llm_system_prompt
        self.model = load_turbomind_model(model_path, llm_system_prompt, cache_max_entry_count)
        self.gen_config = GenerationConfig(top_p=0.8,
                                top_k=40,
                                temperature=0.8,
                                max_new_tokens=2048,
                                repetition_penalty=1.05)


    def _call(self, prompt : str, stop: Optional[List[str]] = None, **kwargs: Any):
        response = self.model([prompt])
        return response[0].text
    
    def stream(self, prompt: str) -> Iterator[str]:
        ## OpenAI 格式输
        messages = [{'role': 'user', 'content': f'{prompt}'}]
        for response in self.model.stream_infer(messages, gen_config=self.gen_config):
            yield response.text
        
    @property
    def _llm_type(self) -> str:
        return "InternLM2"

@st.cache_resource
def load_turbomind_model(model_dir, system_prompt, cache_max_entry_count):  # hf awq

    logging.info(f"正在从本地:{model_dir}加载模型...")

    model_format = "hf"
    if Path(model_dir).stem.endswith("-4bit"):
        model_format = "awq"

    # model_dir = snapshot_download(model_dir, revision="master", cache_dir="./models")
    

    backend_config = TurbomindEngineConfig(
        model_format=model_format, session_len=32768, cache_max_entry_count=cache_max_entry_count, 
    )

    pipe = pipeline(model_dir, backend_config=backend_config, log_level="ERROR", model_name="internlm2",
                chat_template_config=ChatTemplateConfig('customized_model', meta_instruction=system_prompt) )

    logging.info("完成本地模型的加载") 
    return pipe


@hydra.main(version_base=None, config_path="../configs", config_name="model_cfg")
def test_demo(config):
    model_dir = config.llm_model
    ## download model from modelscope
    if not os.path.exists(model_dir):
        download_model(llm_model_path = model_dir)

    system_prompt = config.llm_system_prompt
    cache_max_entry_count = config.cache_max_entry_count #lmdeploy 4bit,  k/v cache内存占比调整为总显存的 20%
    question="""黑神话悟空发售时间和团队？"""
    if config.use_lmdepoly:
        ## lmdepoly inference
        ## OpenAI 格式输
        messages = [{'role': 'user', 'content': f'{question}'}]
        gen_config = GenerationConfig(top_p=0.8,
                    top_k=40,
                    temperature=0.8,
                    max_new_tokens=2048,
                    repetition_penalty=1.05)
        pipe = load_turbomind_model(model_dir, system_prompt, cache_max_entry_count)
        for response in pipe.stream_infer(messages, gen_config=gen_config):
            print(response.text, end='')
        # response = pipe(['你是谁呀', '介绍下你自己', 'Are you developed by LMDeploy?', '黑神话悟空发售时间和团队？'])
    else:
        ## normal inference
        assert not str(model_dir).endswith("w4a16-4bit"), f"{model_dir} must use lmdeploy inference"
        from rag.simple_rag import InternLM
        base_mode = InternLM(model_path=model_dir, llm_system_prompt=system_prompt)
        # 流式显示, used streaming result
        if config.stream_response:
            logging.info("Streaming response:")
            for chunk in base_mode.stream(question):
                print(chunk, end='', flush=True)
            print("\n")
        # 一次性显示结果
        else:
            response = base_mode(question)
            logging.info(f"question: {question}\n wulewule answer:\n{response}")

if __name__ == "__main__":
    test_demo()