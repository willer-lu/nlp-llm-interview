# 自然语言处理（大模型）算法工程师面试题目汇总

## 项目简介

此项目创建的初衷是为了帮助人工智能、自然语言处理和大语言模型相关背景的同学找工作使用。项目将汇总并整理各类与自然语言处理（NLP）和大模型相关的面试题目，包括但不限于以下内容：

- 机器学习基础
- 深度学习基础
- 自然语言处理基础
- Transformer、Bert、GPT系列模型
- 大模型架构与大模型预训练
- 大模型微调
- 大模型训练与推理加速
- 检索增强生成
- 大模型评测
- 大模型agent
- pytorch基础
- 其他问题

欢迎大家加入项目的建设和维护，共同完善和丰富题库内容。
目前主要针对文本模态，也欢迎多模态方向的同学加入。
答案来源于外部链接，如有侵权请联系删除，如有链接不可用请及时指出。

## 目录

1. [机器学习基础](#机器学习基础)
2. [深度学习基础](#深度学习基础)
3. [自然语言处理基础](#自然语言处理基础)
4. [Transformer、Bert、GPT系列模型](#transformerbertgpt系列模型)
5. [大模型架构与大模型预训练](#大模型架构与大模型预训练)
6. [大模型微调](#大模型微调)
7. [大模型训练与推理加速](#大模型训练与推理加速)
8. [大模型幻觉](#大模型幻觉)
9. [大模型评测](#大模型评测)
10. [大模型agent](#大模型agent)
11. [其他问题](#其他问题)
12. [pytorch基础](#pytorch基础)
13. [贡献指南](#贡献指南)

## 机器学习基础

### 机器学习基础知识

1. [梯度裁剪目的、原理、示例代码](https://blog.csdn.net/ZacharyGz/article/details/135410610)
2. 样本均衡
3. 不调包实现shuffle
4. 正则化
   

## 深度学习基础

### 模型训练

1. [模型显存占用分析](https://blog.51cto.com/welcomeweb/7213052)
2. [weight_decay](https://blog.csdn.net/zhaohongfei_358/article/details/129625803)
3. [mse损失函数和cross entropy的区别](https://blog.csdn.net/weixin_41888257/article/details/104894141)
4. [输入token太长了怎么办](https://zhuanlan.zhihu.com/p/493424507)
5. [怎么从0搭建一个深度学习模型](https://blog.csdn.net/AAI666666/article/details/135975253)
6. [模型并行训练]
7. dropout
8. 显存不够一般怎么解决
9. 为什么要使用gpu

### 模型推理
1. [模型推理占用的显存】多卡推理]

## 自然语言处理基础

1. 什么是tokenizer？    
    [Tokenizer详解](https://blog.csdn.net/lsb2002/article/details/133095184)    
    [Tokenizer-huggingface实战](https://blog.csdn.net/weixin_50592077/article/details/131597070)
2. [TF_IDF](https://baike.baidu.com/item/tf-idf/8816134?fr=ge_ala)
3. [Bi-LSTM](https://blog.csdn.net/m0_59749089/article/details/128754246)






## Transformer、Bert、GPT系列模型

### Transformer

1. [一文读懂Transformer](https://blog.csdn.net/weixin_42475060/article/details/121101749)
2. [Transformer位置编码](https://blog.csdn.net/xian0710830114/article/details/133377460)
3. [BatchNorm、LayerNorm以及GroupNorm](https://www.bilibili.com/video/BV1UG411f7DL/?spm_id_from=333.999.0.0)
4. [transformers库的基本使用方式](https://blog.csdn.net/pipisorry/article/details/131003691)
5. [safetensor](https://zhuanlan.zhihu.com/p/695555072)

### Bert
1. [BERT详解：概念、原理与应用](https://blog.csdn.net/fuhanghang/article/details/129524848)

### GPT

1. [GPT-1](https://zhuanlan.zhihu.com/p/625184011)
2. [GPT-2](https://blog.csdn.net/2401_85375298/article/details/139419273)
3. [GPT-3、3.5、4](https://zhuanlan.zhihu.com/p/616691512)


### 模型比较

1. [GPT和BERT的区别](https://zhuanlan.zhihu.com/p/709550645)
2. [为什么现在的LLM大多都是decoder-only的架构？](https://blog.csdn.net/qq_36372352/article/details/140237927)
3. 位置编码比较
4. loss比较

## 大模型架构与大模型预训练

### 大模型预训练

1. [NLP分词算法（总结篇）](https://zhuanlan.zhihu.com/p/679127448)
2. [如何缓解LLM复读机问题？](https://blog.csdn.net/aigchouse/article/details/139510919)
  
3. [对强化学习在LLM中的理解](https://zhuanlan.zhihu.com/p/692074916)
4. 半精度训练



### ChatGPT

1. [大语言模型的预训练RLHF](https://blog.csdn.net/sinat_39620217/article/details/131796342)

### llama
1. [llama 3.1结构详解](https://zhuanlan.zhihu.com/p/710780476)
2. [llama2](https://blog.csdn.net/v_JULY_v/article/details/129709105)
3. [llama](https://blog.csdn.net/qq_42533357/article/details/136993941)

### 大模型相关知识
1. 大模型的温度系数，top-p,top-k
2. 模型的输入上下文窗口
3. 如何处理更长的文本
4. prefix decoder和causal decoder的区别
5. 大模型涌现能力

### ChatGLM
1. [chatGLM](https://zhuanlan.zhihu.com/p/686728765)
## 大模型微调




### 传统微调方法

1. 全参数微调的显存占用
2. 多轮对话任务如何微调

### 参数高效的微调方法

1. lora微调的原理
2. p-tuning微调的原理
3. 不同微调方法的区别？什么情况下应该使用什么微调方法？
4. lora训练项目    
   [IEPile：大规模信息抽取语料库](https://github.com/zjunlp/IEPile/blob/main/README_CN.md)
5. peft库

### SFT

1. 什么是SFT？
2. sft指令微调的数据如何构建
3. 灾难性遗忘

### 微调数据集
1. 不同微调方式的数据集格式和获取方式
2. 微调需要多少条数据
3. 有哪些大模型的训练集

## 大模型训练与推理加速

### 大模型加速库

1. 介绍一下deepspeed的三个zero的区别
2. deepspeed的加速原理
3. 介绍一下vllm    
4. [deepspeed库的基本使用方式](https://blog.csdn.net/myTomorrow_better/article/details/138945584)

### 大模型训练加速

### 大模型推理加速
1. k-v cache
2. 不同精度的推理区别



1. 介绍一下MOE

## 大模型幻觉

### 大模型幻觉问题

1. [大模型幻觉产生原因和解决方案](https://zhuanlan.zhihu.com/p/677935286)

### 向量检索

## 大模型评测

### 大模型评测基准

1. 你了解哪些大模型评测基准？
2. 如何评测大模型



## 大模型应用与大模型agent

### 大模型应用
1. langchain


## 其他问题

### git版本控制

1. [git merge和git rebase的区别？](https://blog.csdn.net/weixin_45565886/article/details/133798840)
2. [git 冲突如何解决](https://blog.csdn.net/weixin_44799217/article/details/132013096)

### 例题

## pytorch基础

1.手撕模型训练的基本代码（假设数据已经准备好）

## 贡献指南

欢迎大家通过提交Issue或Pull Request来贡献题目和答案。在贡献之前，请确保您遵循以下指南：

1. 确保提交的题目和答案清晰明了，并且经过验证。
2. 在提交Pull Request之前，请检查是否有重复的题目。
3. 请遵循Markdown格式，并保持项目的一致性。
