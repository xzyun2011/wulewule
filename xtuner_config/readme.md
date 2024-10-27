# 悟了悟了模型训练



## 一、前言

### 增量预训练（Incremental Pretraining）

增量预训练是指在一个已经预训练过的模型基础上，引入新的数据或特定领域的数据来进一步训练模型，旨在使模型适应新的领域或任务。训练数据不需要标注，模型从中学习通用特征和先验知识。

**特点**：

- **数据规模大**：通常需要大量的新数据，这些数据可能与原始预训练数据分布不同。
- **目标**：增强模型的整体能力和适应性，使其能够处理更广泛的任务和数据。
- **应用场景**：适用于模型需要覆盖新的领域或任务时。例如，一个通用的语言模型需要适应医疗领域的文本处理。

### 指令微调（Instruction Fine-tuning）

指令微调通过在特定任务的数据上进一步训练预训练模型，使其在特定任务上表现更好。指令微调特别关注模型对特定指令或任务的理解和执行能力。

**特点**：

- **数据规模小**：通常使用较小的数据集，专注于某个具体任务的数据，如情感分析、文本分类等。
- **目标**：优化模型在特定任务上的表现，使其能够更好地理解和执行特定指令。
- **应用场景**：适用于模型已经具有足够的通用知识，但需要在特定任务上进行优化时。例如，一个通用的语言模型需要在情感分析任务上表现更好。



一般而言，做"领域知识注入"时，使用指令微调会比增量预训练的效率更高，另外一般大模型泛化性已经比较好了，直接加个RAG也能用，所以如果是基于chat模型做些角色扮演、聊天机器人、学习某种风格等简单任务用微调的方式最好。

但如果你有海量的特定领域数据，或者想让模型学会某个行业深入的知识，可以试试看增量预训练。前提还是得有大量高质量的领域数据，可以是未标注过的，因为模型是去学数据的分布（续写能力）。

本模型是基于xtuner训练，基于此前数据制作一栏的数据即可直接训练，数据格式也可以参考官方文档[xtuner数据集格式和loss计算](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_format.md)



## 二、模型训练

### 配置文件

增量预训练和指令微调训练流程都差不多，通过使用不同的配置文件来加载对应模型和数据即可。

一般要修改的就是base模型、数据源、训练超参数，每个参数的意义可以看官方的[config介绍](https://xtuner.readthedocs.io/zh-cn/latest/user_guides/config.html)

**增量预训练**

增量预训练配置文件可以参考 wulewule/xtuner_config/pretrain/internlm2_5-1_8b-chat_pretrain.py ，下面列出几个重要的

```diff
+ pretrained_model_name_or_path = '/root/models/internlm2_5-1_8b-chat' # 修改成自己的模型路径
+ data_dir="/root/wulewule/data"
+ data_files = [f'{data_dir}/heishenghua_pretraining.jsonl'] # 改成自己的数据路径
+ # prompt_template = PROMPT_TEMPLATE.internlm2_chat #增量预训练用不到
+ accumulative_counts = 1 #单卡训练切记改成1，不然模型学不会东西，切记切记！！！


## qlora的配置
model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))
+ #另外把两处带shuffle的改成False了，想尽量保持原始顺序去学习数据分布
```

**指令微调**

指令微调配置文件可以参考 wulewule/xtuner_config/finetune/internlm2_5_chat_1_8b_qlora_wulewule_all_test.py，下面列出几个重要的

```diff
+ pretrained_model_name_or_path = 'pretrain/merged_internlm2_5-1_8b-chat_pretrain' # 修改成自己的模型路径
+ data_dir="/root/wulewule/data"
+ data_files = [f'{data_dir}/self_cognition_100.jsonl',  f'{data_dir}/huixiangdou_conversations.jsonl'] # 改成自己的数据路径
+ prompt_template = PROMPT_TEMPLATE.internlm2_chat #internlm2_chat默认的
+ accumulative_counts = 1 #单卡训练切记改成1，不然模型学不会东西，切记切记！！！


## qlora的配置
model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))
+ SYSTEM = "你是悟了悟了，由xzyun2011开发的AI助手，专注于回答和《黑神话：悟空》这款游戏相关的问题，你想帮助玩家了解更多这款游戏背后的故事和文化知识。\n" #修改为自己制作数据集时的prompt
```

至于学习率等超参数，需要自己多调试，就经验而言学习了一般设置量级1e-5左右，epoch轮数最好10以下，尽量避免灾难性遗忘问题；

### 模型训练

使用了deepspeed加速训练

```
#增量预训练
xtuner train ./xtuner_config/pretrain/internlm2_5-1_8b-chat_pretrain.py  --work-dir ./pretrain --deepspeed deepspeed_zero1

#指令微调
xtuner train ./xtuner_config/finetune/internlm2_5_chat_1_8b_qlora_wulewule_all_test.py  --work-dir ./finetune --deepspeed deepspeed_zero1
```

loss可视化，读取对应目录tensorboard数据，打开http://localhost:6006/

```
#指令微调为例
tensorboard --logdir ./finetune/work_dirs/internlm2_5_chat_1_8b_qlora_wulewule_all_test.py/20241010_230522/vis_data
```

### 模型转换 + LoRA 合并

在训练完成后，我们会得到几个 `.pth` 文件，这些文件存储了 QLoRA 算法训练过程所更新的参数，而**不是**模型的全部参数。因此我们需要将这些 `.pth` 文件转换为 HuggingFace 格式，并合并入原始的语言模型权重中。

#### 模型转换

XTuner 已经集成好了将模型转换为 HuggingFace 格式的工具，我们只需要执行，pth_file改为自己需要转换的LoRA模型

```
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

## 需要看loss曲线和测试集表现，选择对应的模型
##指令微调为例子，先获取最后保存的一个pth文件
pth_file=`ls -t ./finetune/internlm2_5_chat_1_8b_qlora_wulewule_all_test.py/*.pth | head -n 1| sed 's/:$//' `
# 转换格式
xtuner convert pth_to_hf ./internlm2_5_chat_1_8b_qlora_wulewule_all_test.py ${pth_file} ./hf
```

#### LoRA 合并

XTuner 也已经集成好了合并 LoRA 权重的工具，我们只需执行如下指令

```
# 合并参数
xtuner convert merge /root/models/internlm2_5-1_8b-chat ./hf /root/wulewule/models/wulewule_v1_1_8b --max-shard-size 2GB
```

与转换命令类似，该条合并参数命令会读取原始参数路径 `/root/models/internlm2_5-1_8b-chat` 以及转换为 hf 格式的部分参数路径 `./h`，将两部分参数合并后保存于 `/root/wulewule/models/wulewule_v1_1_8b`，其中每个参数切片的最大文件大小为 2GB。

### 与模型对话

在合并权重后，为了更好地体会到模型的能力，XTuner 也集成了与模型对话的工具。通过如下命令，便可以启动一个与模型对话的简易 Demo。

```
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner chat ./merged --prompt-template internlm2_chat --system "your system"
```

其中 `./merged` 是合并后的权重路径，`--prompt-template internlm2_chat` 指定了对话模板为 InternLM2-Chat，`--system` 则是指定了与模型对话时的 System Prompt，保持和数据训练时一致。
