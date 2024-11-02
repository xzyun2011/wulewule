import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import streamlit as st
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
from download_models import download_model


@st.cache_resource
def load_simple_rag(config, used_lmdeploy=False):
    ## load config
    data_source_dir = config["data_source_dir"]
    db_persist_directory = config["db_persist_directory"]
    llm_model = config["llm_model"]
    embeddings_model = config["embeddings_model"]
    reranker_model = config["reranker_model"]
    llm_system_prompt = config["llm_system_prompt"]
    rag_prompt_template = config["rag_prompt_template"]
    from rag.simple_rag import  WuleRAG

    if not used_lmdeploy:
        from rag.simple_rag import InternLM, WuleRAG
        base_mode = InternLM(model_path=llm_model, llm_system_prompt=llm_system_prompt)
    else:
        from deploy.lmdeploy_model import LmdeployLM, GenerationConfig
        cache_max_entry_count = config.get("cache_max_entry_count", 0.2)
        base_mode = LmdeployLM(model_path=llm_model, llm_system_prompt=llm_system_prompt, cache_max_entry_count=cache_max_entry_count)
    
    ## loda final rag model
    wulewule_rag = WuleRAG(data_source_dir, db_persist_directory, base_mode, embeddings_model, reranker_model, rag_prompt_template)
    return wulewule_rag

GlobalHydra.instance().clear()
@hydra.main(version_base=None, config_path="./configs", config_name="model_cfg")
def main(cfg):
    # omegaconf.dictcfg.DictConfig 转换为普通字典
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    ## download model from modelscope
    if not os.path.exists(config_dict["llm_model"]):
        download_model(llm_model_path =config_dict["llm_model"])

    if cfg.use_rag:
        ## load rag model
        wulewule_model = load_simple_rag(config_dict, used_lmdeploy=cfg.use_lmdepoly)
    elif ( cfg.use_lmdepoly):
        ## load lmdeploy model
        from deploy.lmdeploy_model import load_turbomind_model, GenerationConfig
        wulewule_model = load_turbomind_model(config_dict["llm_model"], config_dict["llm_system_prompt"], config_dict["cache_max_entry_count"])

    ## streamlit setting
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # 在侧边栏中创建一个标题和一个链接
    with st.sidebar:
        st.markdown("## 悟了悟了💡")
        logo_path = "assets/sd_wulewule.webp"
        if os.path.exists(logo_path):
            image = Image.open(logo_path)
            st.image(image, caption='wulewule')
        "[InternLM](https://github.com/InternLM)"
        "[悟了悟了](https://github.com/xzyun2011/wulewule.git)"


    # 创建一个标题
    st.title("悟了悟了：黑神话悟空AI助手🐒")

    # 遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message("user").write(msg["user"])
        st.chat_message("assistant").write(msg["assistant"])

    # Get user input
    if prompt := st.chat_input("请输入你的问题，换行使用Shfit+Enter。"):
        # Display user input
        st.chat_message("user").write(prompt)
        # 流式显示, used streaming result
        if cfg.stream_response:
            # rag
            ## 初始化完整的回答字符串
            full_answer = ""
            with st.chat_message('robot'):
                message_placeholder = st.empty()
                if cfg.use_rag:
                    for cur_response in wulewule_model.query_stream(prompt):
                        full_answer += cur_response
                        # Display robot response in chat message container
                        message_placeholder.markdown(full_answer + '▌')
                elif cfg.use_lmdepoly:
                    # gen_config = GenerationConfig(top_p=0.8,
                    #             top_k=40,
                    #             temperature=0.8,
                    #             max_new_tokens=2048,
                    #             repetition_penalty=1.05)
                    messages = [{'role': 'user', 'content': f'{prompt}'}]
                    for response in wulewule_model.stream_infer(messages):
                        full_answer += response.text
                        # Display robot response in chat message container
                        message_placeholder.markdown(full_answer + '▌')

                message_placeholder.markdown(full_answer)
        # 一次性显示结果
        else:
            if cfg.use_lmdepoly:
                    messages = [{'role': 'user', 'content': f'{prompt}'}]
                    full_answer = wulewule_model(messages).text
            elif cfg.use_rag:         
                full_answer = wulewule_model.query(prompt)
            # 显示回答
            st.chat_message("assistant").write(full_answer)

        # 将问答结果添加到 session_state 的消息历史中
        st.session_state.messages.append({"user": prompt, "assistant": full_answer})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()