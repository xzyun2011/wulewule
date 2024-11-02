from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import json
import argparse

def save_json_once(data, root_path):
    if not os.path.exists(os.path.dirname(root_path)):
        os.makedirs(os.path.dirname(root_path), exist_ok=True)
    with open(root_path, 'at', encoding='utf-8') as f:
        ## 不加 indent，单条数据就是1行
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        # f.write(json.dumps(data, ensure_ascii=False, indent=4) + '\n')


def chunk_files(root_path, save_path, chunk_size = 1024, chunk_overlap = 50):
    if os.path.isfile(root_path):
        print(f"Start loading txt files: {root_path}")
        loader = TextLoader(root_path, encoding='utf-8',autodetect_encoding=True)
    elif os.path.isdir(root_path):
        print(f"Start loading dir: {root_path}")
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(root_path, glob="*.txt", show_progress=True,
                                loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    else:
        raise ValueError(f"'{root_path}' 不存在。")


    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    ## 中文文档优先ChineseRecursiveTextSplitter https://github.com/chatchat-space/Langchain-Chatchat/blob/master/text_splitter/chinese_recursive_text_splitter.py
    ##英文的优先RecursiveCharacterTextSplitter
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
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print(f"Start splitting txt files...")
    texts = text_splitter.split_documents(documents)


    print(f"Chunk-size: {chunk_size}, start saving chunked texts in json...")
    """
    XTuner 定义的增量预训练数据格式准备自定义数据:
    [
        {
            "conversation":[
                {
                    "input": "",
                    "output": "xxx"
                },
            ]
        },
        {
            "conversation":[
                {
                    "input": "",
                    "output": "xxx"
                },
            ]
        }
    ]
    """
    for index, doc in tqdm(enumerate(texts), total=len(texts), desc="Saving JSON files"):
        data = {
            "conversation":[
                {
                    "input": "",
                    "output": f"{doc.page_content}"
                },
            ]
        }
        save_json_once(data, save_path)
    print(f"Done, conversations saved in {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate self cognition dataset')
    parser.add_argument('--root-path', type=str, default="./", help='original data file/dir path')
    parser.add_argument('--save-path', type=str, default="./incremental_pretraining.jsonl", help='json file save path')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Maximum number of characters that a chunk can contain')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Overlap characters between two adjacent chunks')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    root_path=args.root_path
    save_path=args.save_path
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    chunk_files(root_path, save_path, chunk_size, chunk_overlap)

if __name__ == '__main__':
    main()
