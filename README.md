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

1. 梯度裁剪

   [梯度裁剪目的、原理、示例代码](https://blog.csdn.net/ZacharyGz/article/details/135410610)
   

## 深度学习基础

### 模型训练

1. 模型训练所占用的显存由哪些部分组成？
2. [weight_decay](https://blog.csdn.net/zhaohongfei_358/article/details/129625803)
3. [mse损失函数和cross entropy的区别](https://blog.csdn.net/weixin_41888257/article/details/104894141)
4. [输入token太长了怎么办](https://zhuanlan.zhihu.com/p/493424507)

## 自然语言处理基础

1. 什么是tokenizer？    
    [Tokenizer详解](https://blog.csdn.net/lsb2002/article/details/133095184)    
    [Tokenizer-huggingface实战](https://blog.csdn.net/weixin_50592077/article/details/131597070)
2. 介绍一下TF-IDF    
    [TF_IDF](https://baike.baidu.com/item/tf-idf/8816134?fr=ge_ala)
3. [Bi-LSTM](https://blog.csdn.net/m0_59749089/article/details/128754246)






## Transformer、Bert、GPT系列模型

### Transformer

1. 介绍Transformer

   [一文读懂Transformer](https://blog.csdn.net/weixin_42475060/article/details/121101749)
2. Transformer位置编码

   [Transformer位置编码](https://blog.csdn.net/xian0710830114/article/details/133377460)
3. 为什么Transformer要用LayerNorm
4. BatchNorm、LayerNorm以及GroupNorm的区别

   [BatchNorm、LayerNorm以及GroupNorm](https://www.bilibili.com/video/BV1UG411f7DL/?spm_id_from=333.999.0.0)
5. [transformers库的基本使用方式](https://blog.csdn.net/pipisorry/article/details/131003691)

### Bert
1. [BERT详解：概念、原理与应用](https://blog.csdn.net/fuhanghang/article/details/129524848)

### GPT

1. GPT的结构
2. GPT的预训练任务和预训练方式
3. GPT在下游任务的微调方法

### 模型比较

1. 简述GPT和BERT的区别

   <details>
      <summary>查看答案</summary>
      <pre>
   BERT:双向 预训练语言模型+fine-tuning(微调)
   GPT:自回归 预训练语言模型+prompting(指示/提示)
   BERT和GPT是近年来自然语言处理领域中非常重要的模型，它们代表了现代NLP技术的发展。
   应用上的差别:
   BERT主要用于自然语言理解，具体应用如下:
          问答系统:BERT可以在问答系统中用来理解问题并生成答案。
          句子相似度比较:BERT可以用来比较两个句子之间的相似程度
          文本分类:BERT可以用来对文本进行分类。
          情感分析:BERT可以用来对文本进行情感分析。
          命名实体识别:BERT可以用来识别文本中的命名实体。
   GPT在文本生成方面表现尤为优秀，其主要具体应用如下:
          文本生成:GPT可以用来生成文本。
          文本自动完成:GPT可以用来自动完成用户输入的文本。
          语言翻译:GPT可以用来生成翻译后的文本。
          对话生成: GPT可以用来生成对话
          摘要生成: GPT可以用来生成文章摘要
   预训练的区别：
        	在Bert的预训练中，主要是用完形填空的方式补全随机mask的内容。
          在GPT的预训练中，主要是预测下一个token。
   使用方法的差别:
   BERT:fine-tuning(微调)。微调是指模型要做某个专业领域任务时，需要收集相关的专业领域数据，做模型的小幅调整，更新相关参数。
   GPT:prompting(提示工程)。prompt是指当模型要做某个专业领域的任务时，我提供给他一些示例、或者引导。但不用更新模型参数。
      </pre>
      </details>
2. 为什么现在的LLM大多都是decoder-only的架构？

   <details>
      <summary>查看答案</summary>
      <pre>
   LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。
   所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。目前AI大模型的架构基本都采用了Decoder-only(仅解码器)，这一类架构的优势在于可以更容易完成文本生成任务，主流大模型如OpenAl开发的GPT系列、英伟达开发的Megatron-LM等均是采用此架构。另外，研究者们发现模型在增大参数量之后，尤其是加上指令微调之后，其他架构能做的Decoder-only模型也都能做了(比如一系列NLU任务)，同时还有更高的上限和回复多样性。
      </pre>
      </details>

## 大模型架构与大模型预训练

### 大模型预训练

1. [NLP分词算法（总结篇）](https://zhuanlan.zhihu.com/p/679127448)
2. [如何缓解LLM复读机问题？](https://blog.csdn.net/aigchouse/article/details/139510919)
  
3. 谈一谈你对强化学习在LLM中的理解
4. 分布式训练
数据并行、模型并行
单机多卡、多机多卡

### ChatGPT

1. 介绍一下ChatGPT的训练方式（RLHF）
   [大语言模型的预训练[2]:GPT、GPT2、GPT3、GPT3.5、GPT4相关理论知识和模型实现]（https://blog.csdn.net/sinat_39620217/article/details/131796342）

### llama
1. [llama 3.1结构详解](https://zhuanlan.zhihu.com/p/710780476)
2. [llama2](https://blog.csdn.net/v_JULY_v/article/details/129709105)
3. [llama](https://blog.csdn.net/qq_42533357/article/details/136993941)
## 大模型微调

### 参数高效的微调方法

1. lora微调的原理
2. p-tuning微调的原理
3. 不同微调方法的区别？什么情况下应该使用什么微调方法？
4. lora训练项目
   [IEPile：大规模信息抽取语料库](https://github.com/zjunlp/IEPile/blob/main/README_CN.md)

### SFT

1. 什么是SFT？

## 大模型训练与推理加速

### 大模型加速库

1. 介绍一下deepspeed的三个zero的区别
2. deepspeed的加速原理
3. 介绍一下vllm
4. [deepspeed库的基本使用方式](https://blog.csdn.net/myTomorrow_better/article/details/138945584)

### 大模型训练加速

### 大模型推理加速



1. 介绍一下MOE

## 大模型幻觉

### 大模型幻觉问题

1. 什么是大模型幻觉？

### 向量检索

## 大模型评测

### 大模型评测基准

1. 你了解哪些大模型评测基准？

### 例题

## 大模型agent

这一部分将整理关于大模型agent相关的题目。

### 例题

## 其他问题

### git版本控制

1. git merge和git rebase的区别？
2. 发生冲突怎么办？

### 例题

## pytorch基础

1.手撕模型训练的基本代码（假设数据已经准备好）

## 贡献指南

欢迎大家通过提交Issue或Pull Request来贡献题目和答案。在贡献之前，请确保您遵循以下指南：

1. 确保提交的题目和答案清晰明了，并且经过验证。
2. 在提交Pull Request之前，请检查是否有重复的题目。
3. 请遵循Markdown格式，并保持项目的一致性。
