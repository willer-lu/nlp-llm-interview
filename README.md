# 自然语言处理（大模型）算法工程师面试题目汇总(知识点总结)

## 项目简介

此项目创建的初衷是为了帮助人工智能、自然语言处理和大语言模型相关背景的同学找工作使用。项目将汇总并整理各类与自然语言处理（NLP）和大模型相关的面试题目(知识点)，包括但不限于以下内容：

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
10. [大模型应用与大模型agent](#大模型应用与大模型agent)
11. [其他问题](#其他问题)
12. [pytorch基础](#pytorch基础)
13. [贡献指南](#贡献指南)

## 机器学习基础

### 机器学习基础知识

1. [梯度裁剪目的、原理、示例代码](https://blog.csdn.net/ZacharyGz/article/details/135410610)
2. [样本不均衡问题的解决](https://zhuanlan.zhihu.com/p/466994436)
3. [机器学习正则化](https://blog.csdn.net/qq_31267769/article/details/103567455)
4. 如何解决测试集和验证集分布差异问题
5. 机器学习的分类（有监督无监督）
6. 逻辑回归损失函数为什么用交叉熵不用MSE？
7. MHA中的dropout和MLP的dropout有什么区别？
8. 针对小样本的处理方式
9. 交叉熵损失的推导方式
10. 集成学习

## 深度学习基础

### 模型训练

1. [模型显存占用分析](https://blog.51cto.com/welcomeweb/7213052)
2. [weight_decay](https://blog.csdn.net/zhaohongfei_358/article/details/129625803)
3. [mse损失函数和cross entropy的区别](https://blog.csdn.net/weixin_41888257/article/details/104894141)
4. [输入token太长了怎么办](https://zhuanlan.zhihu.com/p/493424507)
5. [怎么从0搭建一个深度学习模型](https://blog.csdn.net/AAI666666/article/details/135975253)
6. [模型并行训练](https://blog.csdn.net/v_JULY_v/article/details/132462452)
7. [大模型训练显存不够一般怎么解决](https://zhuanlan.zhihu.com/p/693191199)
8. [gpu对深度学习的作用](https://www.zhihu.com/tardis/bd/art/106669828)
9. 学习率 warmup
10. 数据清洗策略
11. DP和DDP的区别
12. 不同优化器的区别

### 模型推理

1. [模型推理占用的显存](https://blog.csdn.net/jiaxin576/article/details/139276270)
2. [DataParallel多卡推理](https://blog.51cto.com/u_16213591/10327981)

### 模型结构、损失函数

1. 残差网络
2. L1 loss和L2 loss的区别

## 自然语言处理基础

1. [Tokenizer详解](https://blog.csdn.net/lsb2002/article/details/133095184)[Tokenizer-huggingface实战](https://blog.csdn.net/weixin_50592077/article/details/131597070)
2. [TF_IDF](https://baike.baidu.com/item/tf-idf/8816134?fr=ge_ala)
3. [Bi-LSTM](https://blog.csdn.net/m0_59749089/article/details/128754246)
4. RMSNorm
5. 独热编码和embedding的用途，各自优势，为什么用
6. word2vec为什么只用一个隐藏层
7. CNN处理图像
8. bpe分词
9. 知道多个词向量怎么表示出整个句子，说出用模型和不用模型的方法
10. 多个不同长度的句子怎么转为相同长度的embedding，说出用模型和不用模型的方法

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

### T5

### 模型比较

1. [GPT和BERT的区别](https://zhuanlan.zhihu.com/p/709550645)
2. [为什么现在的LLM大多都是decoder-only的架构？](https://blog.csdn.net/qq_36372352/article/details/140237927)
3. [不同位置编码的分析](https://www.zhihu.com/tardis/bd/art/675243992?source_id=1001)

## 大模型架构与大模型预训练

### 大模型预训练

1. [NLP分词算法（总结篇）](https://zhuanlan.zhihu.com/p/679127448)
2. [如何缓解LLM复读机问题？](https://blog.csdn.net/aigchouse/article/details/139510919)
3. [对强化学习在LLM中的理解](https://zhuanlan.zhihu.com/p/692074916)
4. [大模型量化](https://zhuanlan.zhihu.com/p/649460612)
5. 大模型分布式训练
6. 如何获得高质量的训练数据
7. [大语言模型模型对齐的方法](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247562456&idx=3&sn=a4f80efb78ae8c0926332ae52c253259&chksm=ea63e859903cb4571ca4d05d7f624e32304eaad10be8a8b3ce47e427103f77f5e372ac2eb51a&scene=27)
8. flash attention
9. group query attention
10. page attention
11. 模型训练速度受哪些方面影响
12. 偏好对齐的理解，有哪些方式
13. 训练精度
14. 什么是scale law
15. 如何评估训练数据的质量（定量定性）
16. 大模型训练过程学习率一般如何变化的，退火阶段学习率如何变化的

### ChatGPT

1. [大语言模型的预训练RLHF](https://blog.csdn.net/sinat_39620217/article/details/131796342)

### llama

1. [llama 3.1结构详解](https://zhuanlan.zhihu.com/p/710780476)
2. [llama2](https://blog.csdn.net/v_JULY_v/article/details/129709105)
3. [llama](https://blog.csdn.net/qq_42533357/article/details/136993941)

### ChatGLM

1. [chatGLM](https://zhuanlan.zhihu.com/p/686728765)
2. chatGLM2
3. chatGLM3

### 其他大模型
1. 百川
2. 千问
3. 羊驼

### 长文本大模型技术
1. 提示压缩技术
2. 长文本技术策略
3. long lora

### 其他大模型相关知识

1. [大模型的温度系数](https://zhuanlan.zhihu.com/p/666670367)
2. [大模型上下文窗口](https://blog.csdn.net/sinat_37574187/article/details/140515743)
3. [如何处理更长的文本](https://blog.csdn.net/weixin_46103454/article/details/140334541)
4. [prefix decoder和causal decoder的区别](https://zhuanlan.zhihu.com/p/704345908)
5. [大模型涌现能力](https://zhuanlan.zhihu.com/p/621438653)
6. [大模型若干参数的解析](https://blog.csdn.net/u012856866/article/details/140308083)
7. 大模型的性能和哪些因素有关
8. encoder-decoder分离有什么好处
9. 稀疏注意力机制
10. 大模型提示学习
11. 提升推理能力和指令跟随能力哪个更难，为什么，提升指令跟随能力的优化方式和其他的比如推理有什么不一样的地方



## 大模型微调
1. 微调了还有必要使用prompt吗
### 传统微调方法

1. 全参数微调的显存占用、训练时间
2. 多轮对话任务如何微调

### 参数高效的微调方法

1. [lora微调的原理](https://zhuanlan.zhihu.com/p/702629428)
2. [p-tuning微调的原理](https://zhuanlan.zhihu.com/p/635848732)
3. [不同微调方法的总结](https://blog.csdn.net/lvaolan123456/article/details/139809203)
4. [lora训练项目--IEPile：大规模信息抽取语料库](https://github.com/zjunlp/IEPile/blob/main/README_CN.md)
5. [peft库详解](https://blog.csdn.net/qq_41185868/article/details/131831931)
6. Qlora
7. 模型采用的MLP的adapter，q former adapter，它们之间的一个选型的区别是什么，以及各自的优势是什么？



### SFT

1. [什么是SFT？](https://blog.csdn.net/sunyuhua_keyboard/article/details/140096441)
2. [sft指令微调的数据如何构建](https://blog.csdn.net/Code1994/article/details/140922301)
3. [灾难性遗忘-增量预训练方法](https://blog.csdn.net/2401_85375186/article/details/140669768)

### 微调数据集

1. 不同微调方式的数据集格式和获取方式
2. 微调需要多少条数据
3. 有哪些大模型的训练集

### 强化学习
1. DPO算法
2. PPO算法
3. dpo训完了一般输出长度会变化吗？如何解决这个问题

## 大模型训练与推理加速

1. [混合专家系统MOE](https://huggingface.co/blog/zh/moe)

### 大模型加速库

1. [介绍一下deepspeed的三个zero的区别](https://blog.csdn.net/baoyan2015/article/details/136820078)
3. [deepspeed的加速原理](https://blog.csdn.net/zwqjoy/article/details/130732601)
4. [介绍一下vllm](https://www.lixueduan.com/posts/ai/03-inference-vllm/)
5. [deepspeed库的基本使用方式](https://blog.csdn.net/myTomorrow_better/article/details/138945584)

### cuda相关知识
1.  SM，SP，warp 相关概念和关系
2.  cuda加速原理
3.  hopper架构

### 大模型训练加速
1. 混合并行策略是否了解，介绍一下 TP，PP，DP 的使用场景
2. 介绍一下混合精度训练的原理和优势
3. 量化技术
4. 对于数据并行的理解

### 大模型推理加速

1. [k-v cache](https://www.jianshu.com/p/22daf73f5c9a)
2. [大语言模型LLM基础：推理/不同模型/量化对显存、推理速度和性能的影响](https://blog.csdn.net/weixin_45498383/article/details/140058934)

## 大模型幻觉


### 大模型安全
1. 大模型安全包含哪些方面的内容，有量化指标吗
2. 有什么防护措施

### 大模型幻觉问题

1. [大模型幻觉产生原因和解决方案](https://zhuanlan.zhihu.com/p/677935286)

### 检索增强生成

1. 检索增强生成算法介绍
2. 有哪些检索方式
3. RAG 长记忆检索的时候怎么提高召回率的？
4. 文本向量模型BGE


## 大模型评测

### 大模型评测基准

1. 你了解哪些大模型评测基准？
2. 如何评测大模型
3. 指令跟随能力的评估集有什么，如何评估的？
4. gsm8k和math评估集有什么区别？
5. mbpp和hella swag评估集有什么区别？

## 大模型应用与大模型agent

### 大模型应用

1. langchain
2. huggingface TGI
3. 大模型函数调用能力如何训练或微调、怎么组织文本的格式喂给模型、怎么把下游的一些工具/插件变成模型可以理解的方式？
4. 大模型和推荐有什么结合方式？

## 其他问题

### git版本控制

1. [git merge和git rebase的区别？](https://blog.csdn.net/weixin_45565886/article/details/133798840)
2. [git 冲突如何解决](https://blog.csdn.net/weixin_44799217/article/details/132013096)

### 其他计算机知识
1. 流水线并行

### python基础
1. python的迭代器和生成器的区别
2. 装饰器、*args、**kwargs

## pytorch基础
- 注：可能会在面试过程中考察手撕代码，包括但不限于基本训练代码（MLP+梯度更新）、normalization、经典模型的forward、损失函数
1. torchrun命令
2. 

## 贡献指南

欢迎大家通过提交Issue或Pull Request来贡献题目和答案。在贡献之前，请确保您遵循以下指南：

1. 确保提交的题目和答案清晰明了，并且经过验证。
2. 在提交Pull Request之前，请检查是否有重复的题目。
3. 请遵循Markdown格式，并保持项目的一致性。
