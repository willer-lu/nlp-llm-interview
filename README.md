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

欢迎大家加入项目的建设和维护，共同完善和丰富题库内容。目前主要针对文本模态，也欢迎多模态方向的同学加入。    
答案来源于多方参考，如有侵权请联系删除。

## 目录

1. [机器学习基础](#机器学习基础)
2. [深度学习基础](#深度学习基础)
3. [自然语言处理基础](#自然语言处理基础)
4. [Transformer、Bert、GPT系列模型](#Transformer、Bert、GPT系列模型)
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



## 深度学习基础


### 模型训练
1.模型训练所占用的显存由哪些部分组成？


## 自然语言处理基础

1. 什么是tokenizer？
2. 介绍一下TF-IDF

## Transformer、Bert、GPT系列模型

### Transformer
1.简单介绍Transformer
      <details>
      <summary>查看答案</summary>
      <pre>
Transformer是一种序列到序列的模型，以下是Transformer的一些重要组成部分和特点：

自注意力机制（Self-Attention）：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。
多头注意力（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。
位置编码（Positional Encoding）：由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
残差连接和层归一化（Residual Connections and Layer Normalization）：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
      </pre>
      </details>
2. BatchNorm和LayerNorm的区别

### Bert
1.Bert的结构      
2.Bert的预训练任务和预训练方式      
3.Bert在下游任务的微调方法      

### GPT
1.GPT的结构      
2.GPT的预训练任务和预训练方式      
3.GPT在下游任务的微调方法      

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
1. 大语言模型中有哪些分词技术？

      <details>
      <summary>查看答案</summary>
      <pre>
   分词是将原始文本转化为一系列较小单位(称为令牌)的过程，这些令牌可以是单词、子词或字符。在大型语言模型中使用的一些分词方法和技术包括:
   基于单词的分词:这种方法将文本分割成单个单词，将每个单词视为一个单独的令牌。虽然简单直观基于单词的分词可能会在处理词汇表之外的单词时遇到问题，并且可能无法有效处理具有复杂形态的语言。
   基于子词的分词:基于子词的方法，例如字节对编码(Byte Pair Encoding，BPE)和WordPiece，将文本分割成可以组合成整个单词的较小单元。这种方法使LLMs能够处理词汇表之外的单词，并更好地捕捉不同语言的结构。BPE，例如，合并最常出现的字符对以创建子词单元，而WordPiece采用数据驱动的方法将单词分割成子词令牌。
   基于字符的分词:这种方法将单个字符视为令牌。虽然它可以处理任何输入文本，但基于字符的分词通常需要更大的模型和更多的计算资源，因为它需要处理更长的令牌序列。
      </pre>
      </details>

2. 如何缓解LLM复读机问题？

      <details>
      <summary>查看答案</summary>
      <pre>
   多样性训练数据:在训练阶段，尽量使用多样性的语料库来训练模型，避免数据偏差和重复文本的问题。
   引入噪声:在生成文本时，可以引入一些随机性或噪声，例如通过采样不同的词或短语，或者引入随机的变换操作，以增加生成文本的多样性。
   温度参数调整:温度参数是用来控制生成文本的多样性的一个参数。通过调整温度参数的值，可以控制生成文本的独创性和多样性，从而减少复读机问题的出现。
   后处理和过滤:对生成的文本进行后处理和过滤，去除重复的句子或短语，以提高生成文本的质量和多样性。
   Beam搜索调整:在生成文本时，可以调整Beam搜索算法的参数。Beam搜索是一种常用的生成策略，它在生成过程中维护了一个候选序列的集合。通过调整Beam大小和搜索宽度，可以控制生成文本的多样性和创造性。
   人工干预和控制:对于关键任务或敏感场景，可以引入人工干预和控制机制，对生成的文本进行审查和筛选，确保生成结果的准确性和多样性。
      </pre>
      </details>


3. 谈一谈你对强化学习在LLM中的理解

### ChatGPT
1.介绍一下ChatGPT地训练方式（RLHF）


## 大模型微调
### 参数高效的微调方法
1. lora微调的原理      
2. p-tuning微调的原理      
3. 不同微调方法的区别？什么情况下应该使用什么微调方法？
### SFT
1. 什么是SFT？










## 大模型训练与推理加速

### 大模型加速库
1. 介绍一下deepspeed的三个zero的区别
2. deepspeed的加速原理
3. 介绍一下vllm

### 大模型训练加速

### 大模型推理加速
1. 介绍一下MOE







## 大模型幻觉
### 大模型幻觉问题
1.什么是大模型幻觉？


### 向量检索

## 大模型评测

这一部分将整理一些实际项目中的案例分析题目。

### 例题

## 大模型agent

这一部分将整理关于大模型agent相关的题目。

### 例题

## 其他问题

这一部分将整理一些其他工程类的问题，如Linux和git相关的命令等。

### 例题

## pytorch基础
1.手撕模型训练的基本代码（假设数据已经准备好）
## 贡献指南

欢迎大家通过提交Issue或Pull Request来贡献题目和答案。在贡献之前，请确保您遵循以下指南：

1. 确保提交的题目和答案清晰明了，并且经过验证。
2. 在提交Pull Request之前，请检查是否有重复的题目。
3. 请遵循Markdown格式，并保持项目的一致性。

