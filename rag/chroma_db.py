from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os
import logging


def is_chroma_data_exist(directory):
    # 检查目录下是否有 Chroma 数据文件，假设 Chroma 会创建这些文件
    return os.path.exists(os.path.join(directory, "chroma.sqlite3"))


def load_documents(root_path):
    if os.path.isfile(root_path):
        logging.info(f"Start loading txt files: {root_path}")
        loader = TextLoader(root_path, encoding='utf-8',autodetect_encoding=True)
    elif os.path.isdir(root_path):
        logging.info(f"Start loading dir: {root_path}")
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(root_path, glob="*.txt", show_progress=True,
                                loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    else:
        raise ValueError(f"'{root_path}' 不存在。")
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents")
    return documents


def get_split_docs(data_source_dir="/root/wulewule/data"):
        documents = load_documents(data_source_dir)
        #创建文本分割器实例
        ## 中文文档优先ChineseRecursiveTextSplitter https://github.com/chatchat-space/Langchain-Chatchat/blob/master/text_splitter/chinese_recursive_text_splitter.py
        ## 英文的优先RecursiveCharacterTextSplitter
        ## 按字符递归拆分,添加附加标点符号
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                "。",
                " ，",
                ".",
                ",",
                "\u200B",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ""],
            chunk_size=768, chunk_overlap=32)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

def get_chroma_db(data_source_dir, persist_directory, embeddings_model):
    if is_chroma_data_exist(persist_directory):
        # 目录中已有数据，直接加载
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        logging.info(f"loaded disk data")
    else:
        # 目录中没有数据，重新生成并保存
        split_docs = get_split_docs(data_source_dir)
        # 加载数据库
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory=persist_directory)
        vectordb.persist()
    return vectordb

