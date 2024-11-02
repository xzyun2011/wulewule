import os
import torch
import re
import hydra
import logging
from langchain_community.vectorstores import Chroma
from BCEmbedding.tools.langchain import BCERerank
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, List, Optional, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.retrievers import ContextualCompressionRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TextIteratorStreamer
from threading import Thread
from modelscope.hub.snapshot_download import snapshot_download

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chroma_db import get_chroma_db
from download_models import download_model


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename='ragchat.log',
                    filemode='w')

class InternLM(LLM):   
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None
    llm_system_prompt: str=""
    def __init__(self, model_path: str, llm_system_prompt: str):
        super().__init__()
        logging.info(f"正在从本地:{model_path}加载模型...")
        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
            self.model.eval()  # 将模型设置为评估模式
            self.llm_system_prompt = llm_system_prompt
            logging.info("完成本地模型的加载")        
        except Exception as e:
            logging.error(f"加载模型时发生错误: {e}")
            raise

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        
        # 重写调用函数
        system_prompt = self.llm_system_prompt

        messages = [("system", system_prompt)]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
    
    def stream(self, prompt: str) -> Iterator[str]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=2048,
            # do_sample=False,
            # top_k=30,
            # top_p=0.85,
            # temperature=0.7,
            # repetition_penalty=1.1,
            streamer=streamer,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for idx, new_text in enumerate(streamer):
            # 跳过prompt部分
            if idx > 0:
                yield new_text

    @property
    def _llm_type(self) -> str:
        return "InternLM2"

class WuleRAG():
    """
    存储检索问答链的对象 
    """
    def __init__(self, data_source_dir, db_persist_directory, base_mode, embeddings_model, reranker_model, rag_prompt_template):
        # 加载自定义 LLM
        self.llm = base_mode

        # 定义 Embeddings
        ## bce-embedding-base_v1 如果路径不对，则下载默认的模型
        if not os.path.exists(embeddings_model):
            if embeddings_model.endswith("bce-embedding-base_v1"):
                embeddings_model = snapshot_download("maidalun1020/bce-embedding-base_v1", revision='master')
                logging.info(f"bce-embedding model not exist, downloading from modelscope \n save to {embeddings_model}")
            else:
                raise ValueError(f"{embeddings_model} model not exist, please reset or re-download your model.")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"batch_size": 1, "normalize_embeddings": True})
        
        ## bce-reranker-base_v1 如果路径不对，则下载默认的模型
        if not os.path.exists(reranker_model):
            if reranker_model.endswith("bce-reranker-base_v1"):
                reranker_model = snapshot_download("maidalun1020/bce-reranker-base_v1", revision='master')
                logging.info(f"reranker_model model not exist, downloading from modelscope \n save to {reranker_model}")
            else:
                raise ValueError(f"{reranker_model} model not exist, please reset or re-download your model.")
        reranker_args = {'model': reranker_model, 'top_n': 5, 'device': 'cuda', "use_fp16": True}
        self.reranker = BCERerank(**reranker_args)
        vectordb = get_chroma_db(data_source_dir, db_persist_directory, self.embeddings)

        # 创建基础检索器
        # retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 2})
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.6},  search_type="similarity_score_threshold" )

        # 创建上下文压缩检索器
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.retriever
        )
        
        # 定义包含 system prompt 的模板
        self.PROMPT = PromptTemplate(
            template=rag_prompt_template, input_variables=["context", "question"]
        )

        # 创建 RetrievalQA 链，包含自定义 prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff"、"map_reduce"、"refine"、"map_rerank"
            retriever=self.compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.PROMPT}
        )
    
    def query_stream(self, query: str) -> Iterator[str]:
        docs = self.compression_retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = self.PROMPT.format(context=context, question=query)
        return self.llm.stream(prompt)

    def query(self, question):
        """
        调用问答链进行回答，如果没有找到相关文档，则使用模型自身的回答
        #使用示例
        question='黑神话悟空发售时间和团队？主要讲了什么故事？'
        result = self.qa_chain({"query": question})
        print(result["result"])
        """
        if not question:
            return "请提供个有用的问题。"

        try:
            # 使用检索链来获取相关文档
            result = self.qa_chain.invoke({"query": question})         
            # logging.info(f"Get rag res:\n{result}")
            
            if 'result' in result:
                answer = result['result']
                final_answer = re.sub(r'^根据提供的信息，\s?', '', answer, flags=re.M).strip()
                return final_answer
            else:
                logging.error("Error: 'result' field not found in the result.")
                return "悟了悟了目前无法提供答案，请稍后再试。"
        except Exception as e:
            # 打印更详细的错误信息，包括traceback
            import traceback
            logging.error(f"An error occurred: {e}\n{traceback.format_exc()}")
            return "悟了悟了遇到了一些技术问题，正在修复中。"


@hydra.main(version_base=None, config_path="../configs", config_name="rag_cfg")
def main(config):
    data_source_dir = config.data_source_dir
    db_persist_directory = config.db_persist_directory
    llm_model = config.llm_model
    embeddings_model = config.embeddings_model
    reranker_model = config.reranker_model
    llm_system_prompt = config.llm_system_prompt
    rag_prompt_template = config.rag_prompt_template

    ## download model from modelscope
    if not os.path.exists(llm_model):
        download_model(llm_model_path = llm_model)

    base_mode = InternLM(model_path=llm_model, llm_system_prompt=llm_system_prompt)
    # from deploy.lmdeploy_model import LmdeployLM
    # base_mode = LmdeployLM(model_path=llm_model, llm_system_prompt=llm_system_prompt, cache_max_entry_count=0.2)
    wulewule_rag = WuleRAG(data_source_dir, db_persist_directory, base_mode, embeddings_model, reranker_model, rag_prompt_template)
    question="""黑神话悟空发售时间和团队？主要讲了什么故事？"""
    # 流式显示, used streaming result
    if config.stream_response:
        logging.info("Streaming response:")
        for chunk in wulewule_rag.query_stream(question):
            print(chunk, end='', flush=True)
        print("\n")
    # 一次性显示结果
    else:
        response = wulewule_rag.query(question)
        logging.info(f"question: {question}\n wulewule answer:\n{response}")

if __name__ == "__main__":
    main()
    # llm_system_prompt = "你是悟了悟了，由xzyun2011开发的AI助手，专注于回答和《黑神话：悟空》这款游戏相关的问题，你想帮助玩家了解更多这款游戏背后的故事和文化知识。"
    # rag_prompt_template = """系统: 你是悟了悟了，由xzyun2011开发的AI助手，专注于回答和《黑神话：悟空》这款游戏相关的问题，你想帮助玩家了解更多这款游戏背后的故事和文化知识。
    # 人类: {question}
    
    # 助手: 我会根据提供的信息来回答。
    
    # 相关上下文:
    # {context}
    
    # 基于以上信息，我的回答是：
    # """
    # data_source_dir = "/root/wulewule/data"   # txt数据目录
    # db_persist_directory ='/root/wulewule/rag/chroma' # chroma向量库数据目录
    # llm_model = "/root/wulewule/models/wulewule_v1_1_8b"
    # embeddings_model = "/root/share/new_models/maidalun1020/bce-embedding-base_v1"
    # reranker_model = "/root/share/new_models/maidalun1020/bce-reranker-base_v1"