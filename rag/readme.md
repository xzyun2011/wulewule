# 悟了悟了RAG使用



## 一、前言

### RAG一般流程

- 将原始数据切分后向量化，制作成向量数据库
- 对用户输入的问题进行 embedding
- 基于 embedding 结果在向量数据库中进行检索
- 对召回数据重排序（选择和问题更接近的结果）
- 依据用户问题和召回数据生成最后的结果

悟了悟了默认`data`目录为txt数据源目录，开启RAG后，会使用bce-embedding-base_v1自动将`data`目录下的txt数据转为换chroma向量库数据，存放在`rag/chroma `目录下（如果该目录下已有数据库文件，则跳过数据库创建），然后使用bce-reranker-base_v1对检索到的信息重排序后，将问题和上下文一起给模型得到最终输出。`rag/simple_rag.py`里是一个简单的demo，参数配置见`configs/rag_cfg.yaml`。

[LangChain](https://python.langchain.com/docs/concepts/rag/)在这块的工具比较好，各种功能都有，本模型的RAG是基于LangChain进行开发的。



## 二、数据库制作

数据库制作代码在`rag/chroma_db.py`中。首先会将txt文本切分成小块，类似此前的增量预训练数据制作，此部分代码不再赘述。

切分后的文本可以直接使用 langchain_community.vectorstores 中的 Chroma制作向量数据库，并将数据库做一个持久化

```
        # 加载数据库
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory=persist_directory)
        vectordb.persist()  #数据库做持久化
```

另外还有一个[Faiss](https://faiss.ai/)数据库，也是主流使用的。Faiss是一个用于高效相似性搜索和密集向量聚类的库。它包含的算法可以搜索任意大小的向量集。langchain已经整合过FAISS，[FAISS in Langchain](https://python.langchain.com/docs/integrations/vectorstores/faiss)



## 三、rag调用

基于LangChain的RAG实现比较简单，需要一个Embeddings和reranker模型，从数据库中提取和输入问题最相关的材料，再把输入问题和对应材料合在一起（prompt中），统一喂给基础的LLM生成最终的答案，prompt类似如下：

```
'材料：“{}”\n 问题：“{}” \n 请仔细阅读参考材料回答问题。'  
```

具体实现参考`rag/simple_rag.py`，核心部分是如下代码， self.llm 可以换成任意模型或者api接口，只要能输入文本，输出文字结果就行；

```python

class WuleRAG():
    """
    存储检索问答链的对象 
    """
    def __init__(self, data_source_dir, db_persist_directory, base_mode, embeddings_model, reranker_model, rag_prompt_template):
        # 加载自定义 LLM
        self.llm = base_mode

        # 定义 Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"batch_size": 1, "normalize_embeddings": True})
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
```

