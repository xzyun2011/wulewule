

## 目标

### 初版

基于InterLM2+RAG实现英雄联盟背景故事相关的聊天助手，主要包含游戏设定、各大势力背景、各位游戏英雄背景故事等，用户问相关名词或者角色名字，可以告诉用户对应的信息（希望减少幻觉情况），回答的结果符合游戏设定

### 进阶版

* 加入多模态功能，如tts（不同英雄的语音回答问题）、生成角色图像或者数字人
* 使用知识图谱构建数据库



## TODO

### 数据收集

#### LoL相关基础数据

此类数据用于finetune模型，对整个英雄联盟宇宙和游戏有一定先验知识

官方游戏资料网（爬点试试...） https://lol.qq.com/data/info-heros.shtml

英雄联盟wiki（可能需要翻译成中文...） https://leagueoflegends.fandom.com/zh/wiki/%E7%AC%A6%E6%96%87?variant=zh-tw 

github上lol相关数据集:

 https://github.com/CNWindson/LOL_District_Database

https://github.com/monkey-hjy/Spider_LOLStory/blob/master/Data/lol%E6%95%85%E4%BA%8B.txt



#### RAG数据库-背景故事类数据

lol宇宙官方网站 https://universe.leagueoflegends.com/zh_TW/

多玩、九游、巴哈姆特等游戏网 https://home.gamer.com.tw/artwork.php?sn=5050462

lol宇宙背景类视频节目《徐老师讲故事》（拔取字幕）https://space.bilibili.com/16794231?spm_id_from=333.788.0.0

### 数据清洗

这块可能是非常费时间的了，只能人工＋代码搞了...希望，到时候调研下看能不能借助大模型如kimi等处理文本数据

### 数据格式转化

转成InterLM适配的格式，RAG是用csv格式看看

### 技术方案

#### 底座模型选型

InterLM2-chat-1.8b 和 InternLM2-chat-7b试试

InternLM-XComposer-VL-7B 多模态的也看看

#### LoRA微调

用XTunner的QLoRA微调

#### RAG框架

用LMDeploy 量化部署和Lagent框架试试

