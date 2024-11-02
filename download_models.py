import os
from modelscope.hub.snapshot_download import snapshot_download

def download_model(llm_model_path = "/root/test_env/models/wulewule_v1_1_8b", repo_id="xzyun2011"):
    save_dir = os.path.dirname(llm_model_path)
    model_name = os.path.basename(llm_model_path)
    if not os.path.exists(llm_model_path):
        print(f"""===============================================\n
        {llm_model_path} not exist!
        Downloading from modelscope...   """)

        os.system(f"mkdir -p  {save_dir}")
        ## modelscope
        model_dir = snapshot_download(f'{repo_id}/{model_name}', cache_dir= save_dir, revision='v1')
        os.system(f"mv {save_dir}/{repo_id}/{model_name}  {save_dir}/{model_name}")

        print(f"""Finished download {model_name}, save to {save_dir}\n
        ===============================================""")

        ## git lfs太慢了
        # os.system('apt install git')
        # os.system('apt install git-lfs')
        # os.system('git lfs install')
        # os.system(f'git clone https://code.openxlab.org.cn/{repo_id}/wulewule_v1_1_8b.git {save_dir}')
        # os.system(f'cd {save_dir} && git lfs pull')