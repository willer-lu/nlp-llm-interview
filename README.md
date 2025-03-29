# 自然语言处理（大模型）算法工程师面试题目汇总(知识点总结)

## 项目简介

此项目创建的初衷是为了帮助人工智能、自然语言处理和大语言模型相关背景的同学找工作使用。项目将汇总并整理各类与自然语言处理（NLP）和大模型相关的面试题目(八股、知识点)，包括但不限于以下内容：

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
### 除八股外，也需要对自己的项目或研究方向深入准备，此不在本文档的讨论范围内。

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
4. [如何解决测试集和训练集分布差异问题](https://zhuanlan.zhihu.com/p/574303282)
5. [机器学习的分类（有监督无监督）](https://blog.csdn.net/wt334502157/article/details/139661576)
6. [逻辑回归损失函数为什么用交叉熵不用MSE？](https://zhuanlan.zhihu.com/p/453411383)
7. [小样本学习](https://blog.csdn.net/hellozhxy/article/details/131332847)
8. [交叉熵损失的推导方式](https://zhuanlan.zhihu.com/p/349435290)
9. [集成学习](https://baijiahao.baidu.com/s?id=1799441021746509245&wfr=spider&for=pc)
10. [拒绝采样](https://zhuanlan.zhihu.com/p/21453360596)
11. [查全率、查准率等指标的区别](https://blog.csdn.net/weixin_59049646/article/details/137771602)
12. [bagging boosting方法](https://cloud.tencent.com/developer/article/1428832)
13. [gan vae](https://blog.csdn.net/liuweni/article/details/144593953)
14. [偏差和方差](https://blog.csdn.net/stay_foolish12/article/details/89289564)

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
9. [学习率 warmup](https://blog.csdn.net/orangerfun/article/details/120400247)
10. [数据清洗策略](https://blog.51cto.com/u_15444/11032611)
11. [DP和DDP的区别](https://zhuanlan.zhihu.com/p/676142368)
12. [不同优化器的区别](https://blog.51cto.com/u_16099252/10542265)
13. [手撕反向传播](https://blog.51cto.com/u_15240054/10029601)
14. [常见的超参数的设置](https://blog.csdn.net/JiexianYao/article/details/143486610)
15. [模型参数量的计算](https://www.cnblogs.com/future-dream/p/18504417)
16. [梯度消失与梯度爆炸](https://blog.csdn.net/Code1994/article/details/143735999)
17. [批量梯度下降](https://blog.csdn.net/m0_51200050/article/details/140077396)
18. [batchsize的选取到底对训练的效果有怎样的影响](https://blog.csdn.net/liuchenbaidu/article/details/143735040)
19. [梯度检查点](https://zhuanlan.zhihu.com/p/19218662446)

### 模型推理

1. [模型推理占用的显存](https://blog.csdn.net/jiaxin576/article/details/139276270)
2. [DataParallel多卡推理](https://blog.51cto.com/u_16213591/10327981)

### 模型结构、损失函数

1. [残差网络](https://zhuanlan.zhihu.com/p/91385516)
2. [L1 loss和L2 loss的区别](https://blog.csdn.net/weixin_51293984/article/details/137395173)

## 自然语言处理基础

1. [Tokenizer详解](https://blog.csdn.net/lsb2002/article/details/133095184)[Tokenizer-huggingface实战](https://blog.csdn.net/weixin_50592077/article/details/131597070)
2. [TF_IDF](https://baike.baidu.com/item/tf-idf/8816134?fr=ge_ala)
3. [Bi-LSTM](https://blog.csdn.net/m0_59749089/article/details/128754246)
4. [RMSNorm](https://blog.csdn.net/yjw123456/article/details/138139970)
5. [独热编码和embedding的区别](https://blog.csdn.net/yunxiu988622/article/details/105816731/)
6. [word2vec](https://zhuanlan.zhihu.com/p/251348767)
7. [CNN处理文本](https://blog.csdn.net/weixin_43156294/article/details/140910917)
8. [bpe分词](https://zhuanlan.zhihu.com/p/698189993)
9. [不同的采样方法](https://zhuanlan.zhihu.com/p/453286395)
10. [模型解码方式](https://zhuanlan.zhihu.com/p/715728509)
11. [constrained beam search](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247550556&idx=2&sn=759a0fbfc878ed8844e12798701d175a&chksm=ebb73e88dcc0b79e0a21a51059d3859000dca073ea640c5b1b2d7f5f7294fc54f37191723f86&scene=27)

## Transformer、Bert、GPT系列模型

### Transformer

1. [一文读懂Transformer](https://blog.csdn.net/weixin_42475060/article/details/121101749)
2. [Transformer位置编码](https://blog.csdn.net/xian0710830114/article/details/133377460)
3. [BatchNorm、LayerNorm以及GroupNorm](https://www.bilibili.com/video/BV1UG411f7DL/?spm_id_from=333.999.0.0)
4. [transformers库的基本使用方式](https://blog.csdn.net/pipisorry/article/details/131003691)
5. [safetensor](https://zhuanlan.zhihu.com/p/695555072)
6. [位置编码汇总](https://blog.csdn.net/qq_45791939/article/details/146075127)

### Bert

1. [BERT详解：概念、原理与应用](https://blog.csdn.net/fuhanghang/article/details/129524848)

### GPT

1. [GPT-1](https://zhuanlan.zhihu.com/p/625184011)
2. [GPT-2](https://blog.csdn.net/2401_85375298/article/details/139419273)
3. [GPT-3、3.5、4](https://zhuanlan.zhihu.com/p/616691512)

### T5
1. [T5模型详解](https://zhuanlan.zhihu.com/p/580554368)
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
5. [大模型分布式训练](https://hub.baai.ac.cn/view/33040)
6. [如何获得高质量的训练数据](https://baijiahao.baidu.com/s?id=1805369548400453514&wfr=spider&for=pc)
7. [大语言模型模型对齐的方法](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247562456&idx=3&sn=a4f80efb78ae8c0926332ae52c253259&chksm=ea63e859903cb4571ca4d05d7f624e32304eaad10be8a8b3ce47e427103f77f5e372ac2eb51a&scene=27)
8. [flash attention](https://zhuanlan.zhihu.com/p/676655352)
9. [group query attention](https://zhuanlan.zhihu.com/p/683370180)
10. [page attention](https://blog.csdn.net/buptgshengod/article/details/132783552)
11. [模型训练速度受哪些方面影响](https://blog.csdn.net/m0_56896669/article/details/137720202)
12. [偏好对齐的理解，有哪些方式](https://zhuanlan.zhihu.com/p/675348061)
13. [什么是scale law](https://zhuanlan.zhihu.com/p/671327709)
14. [大模型训练过程学习率一般如何变化的](https://developer.baidu.com/article/details/2687396)
15. [预训练退火](https://blog.csdn.net/baoyan2015/article/details/136025792)
16. [训练数据来源](https://baijiahao.baidu.com/s?id=1805283661252037515&wfr=spider&for=pc)
17. [训练数据配比](https://www.zhihu.com/question/639096509/answer/3434426739)
18. [持续预训练、后训练\大模型的继续训练](https://blog.csdn.net/aolan123/article/details/144671210)
19. [各种并行策略](https://zhuanlan.zhihu.com/p/17561509307)
20. [序列并行等](https://zhuanlan.zhihu.com/p/659792351)
21. [fp16和bf16的区别](https://zhuanlan.zhihu.com/p/703939408)
22. [高阶矩阵相乘，维度的变化](https://blog.csdn.net/weixin_43135178/article/details/120610743)
### ChatGPT

1. [大语言模型的预训练RLHF](https://blog.csdn.net/sinat_39620217/article/details/131796342)
2. GPT 4v 4o o1等
3. [chatgpt](https://mp.weixin.qq.com/s?__biz=MzU1MzE2NzIzMg==&mid=2247491577&idx=1&sn=9247f63dc89f215ae5cc7bf476ffcec3&chksm=fbf7af16cc802600f182f961ed7ec3afe5910598a426bdb8cb72c93a2a0af7f0fb5961fc882b#rd
)

### llama

1. [llama 3.1结构详解](https://zhuanlan.zhihu.com/p/710780476)
2. [llama2](https://blog.csdn.net/v_JULY_v/article/details/129709105)
3. [llama](https://blog.csdn.net/qq_42533357/article/details/136993941)

### ChatGLM

1. [chatGLM](https://zhuanlan.zhihu.com/p/686728765)
2. [chatGLM2](https://blog.csdn.net/lazycatlove/article/details/140865800)
3. [chatGLM3](https://blog.csdn.net/xiao_ling_yun/article/details/140905424)

### Deep Seek
1. nsa

### 其他大模型（参考官网资料）
1. 百川
2. 千问
3. 羊驼
4. KIMI，文心一言，mistral、gemini、grok、claude

### 长文本大模型技术
1. [提示压缩技术](https://blog.csdn.net/JingYu_365/article/details/141116779)
2. [长文本技术策略](https://zhuanlan.zhihu.com/p/657210829)
3. [long lora](https://zhuanlan.zhihu.com/p/658067243)


### 强化学习相关
1. 策略模型
2. 各种对齐方法汇总

### 其他大模型相关知识

1. [大模型的温度系数](https://zhuanlan.zhihu.com/p/666670367)
2. [大模型上下文窗口](https://blog.csdn.net/sinat_37574187/article/details/140515743)
3. [如何处理更长的文本](https://blog.csdn.net/weixin_46103454/article/details/140334541)
4. [prefix decoder和causal decoder的区别](https://zhuanlan.zhihu.com/p/704345908)
5. [大模型涌现能力](https://zhuanlan.zhihu.com/p/621438653)
6. [大模型若干参数的解析](https://blog.csdn.net/u012856866/article/details/140308083)
7. [大模型的性能和哪些因素有关](https://baijiahao.baidu.com/s?id=1765223469159042766&wfr=spider&for=pc)
8. [稀疏注意力机制](https://zhuanlan.zhihu.com/p/691296437)
9. [大模型提示学习](https://blog.csdn.net/idiotyi/article/details/140263181)
10. [模型压缩与模型蒸馏](https://zhuanlan.zhihu.com/p/638092734)
- 激活函数（GeLU、Swish、GLU）的选择与优化
- 投机采样（Speculative Sampling）与生成策略（Top-k、Top-p、温度系数）
- scaling law
- 大模型知识注入、持续学习（如何避免灾难性遗忘）
- moba
- 如何得到大模型的置信度
- nsp具体的训练方式
- 大模型指令遵循


## 大模型微调
### 传统微调方法

1. [全参数微调的显存占用](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247565072&idx=2&sn=1a94697ed9b5289968e6210214090be0&chksm=eaaa32e01615f923a29195e3109a9456bd6e3076757cea7201c8a8287135af84e5d753d7e96d&scene=27)
2. [多轮对话任务如何微调](https://blog.csdn.net/Code1994/article/details/141156276)

### 参数高效的微调方法

1. [lora微调的原理](https://zhuanlan.zhihu.com/p/702629428)
2. [p-tuning微调的原理](https://zhuanlan.zhihu.com/p/635848732)
3. [不同微调方法的总结](https://blog.csdn.net/lvaolan123456/article/details/139809203)
4. [lora训练项目--IEPile：大规模信息抽取语料库](https://github.com/zjunlp/IEPile/blob/main/README_CN.md)
5. [peft库详解](https://blog.csdn.net/qq_41185868/article/details/131831931)
6. [Qlora](https://zhuanlan.zhihu.com/p/681519805)
7. [adapter](https://zhuanlan.zhihu.com/p/693730345)
8. [微调方法总结](https://blog.csdn.net/2301_82275412/article/details/139009033)
9. PILL（Pluggable Instruction Language Learning）、SSF（Scaling & Shifting Your Features）等其他类型的微调方法
10. llama factory



### SFT

1. [什么是SFT？](https://blog.csdn.net/sunyuhua_keyboard/article/details/140096441)
2. [sft指令微调的数据如何构建](https://blog.csdn.net/Code1994/article/details/140922301)
3. [灾难性遗忘-增量预训练方法](https://blog.csdn.net/2401_85375186/article/details/140669768)

### 微调数据集

1. [微调需要多少条数据](https://blog.csdn.net/2401_85379281/article/details/140314766)
2. [大模型常用的微调数据集](https://blog.csdn.net/weixin_43961909/article/details/138538912)

### 强化学习
1. [DPO算法](https://blog.csdn.net/raelum/article/details/141612193)
2. [PPO算法](https://blog.51cto.com/u_13937572/7929830)
3. [dpo训完了一般输出长度会变化吗？如何解决这个问题](https://www.zhihu.com/question/645365157/answer/3426813979)

## 大模型训练与推理加速

1. [混合专家系统MOE](https://huggingface.co/blog/zh/moe)
2. 梯度累积、激活检查点

### 大模型加速库

1. [介绍一下deepspeed的三个zero的区别](https://blog.csdn.net/baoyan2015/article/details/136820078)
3. [deepspeed的加速原理](https://blog.csdn.net/zwqjoy/article/details/130732601)
4. [介绍一下vllm](https://www.lixueduan.com/posts/ai/03-inference-vllm/)
5. [deepspeed库的基本使用方式](https://blog.csdn.net/myTomorrow_better/article/details/138945584)
6. accelerate 库

### cuda相关知识
1.  [SM，SP，warp 等相关概念和关系](https://zhuanlan.zhihu.com/p/266633373?utm_source=wechat_session&utm_id=0)
2.  [cuda加速原理](http://www.360doc.com/content/24/0429/18/2427359_1121818118.shtml)
3.  [hopper架构](https://zhuanlan.zhihu.com/p/708645371)

### 大模型训练加速
1. [大模型训练并行策略](https://zhuanlan.zhihu.com/p/699944367)
2. [混合精度训练](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247584487&idx=2&sn=52b742d9f714af66b22a4d0a0c50f3b0&chksm=fa88dfc3c6adf6a98cdeab0842b169739a7d9566d3a04357039d30561ea7517e053ae784fe77&scene=27)
3. [量化技术](https://zhuanlan.zhihu.com/p/662881352)
4. [对于数据并行的理解](https://blog.csdn.net/weixin_41204131/article/details/141005134)

### 大模型推理加速

1. [k-v cache](https://www.jianshu.com/p/22daf73f5c9a)
2. [大语言模型LLM基础：推理/不同模型/量化对显存、推理速度和性能的影响](https://blog.csdn.net/weixin_45498383/article/details/140058934)
3. mla

## 大模型幻觉


### 大模型安全
1. [大模型安全包含哪些方面的内容](https://zhuanlan.zhihu.com/p/691384260)
2. [有什么防护措施](https://mp.weixin.qq.com/s?__biz=MzA5MzE5MDAzOA==&mid=2664224181&idx=1&sn=e41362f5710e5e72d87d7bd041102c3e&chksm=8aa419fd987a102f47f02e2b3c90c2b6f40795d5f0feb3ca3edc91bdba77dd3a5cc60be00cf2&scene=27)

### 大模型幻觉问题

1. [大模型幻觉产生原因和解决方案](https://zhuanlan.zhihu.com/p/677935286)
2. 知识注入（以llama2-7b为例）
3. [TaD](https://mp.weixin.qq.com/s/jnD87hrEnrARsCRaL4cmcQ)

### 检索增强生成

1. [检索增强生成算法介绍](https://www.zhihu.com/tardis/bd/art/675509396?source_id=1001)
2. [有哪些检索方式](https://zhuanlan.zhihu.com/p/678657202)
3. [RAG怎么提高召回率的？](https://blog.csdn.net/weixin_45312236/article/details/139120662)
4. [文本向量模型BGE](https://zhuanlan.zhihu.com/p/690856333)
5. - 如何从检索的内容中过滤噪声
- 如何保证生成是基于检索文本的
- 检索文本和内部知识有冲突怎么办
- graphrag
生成式检索
文本向量化、向量嵌入模型
向量的维度
评测的方法和基准




## 大模型评测

### 常见的评测指标
1. bleu、ppl、rouge

### 大模型评测基准

1. [常见的大模型评测基准](https://zhuanlan.zhihu.com/p/710570068)
2. [如何评测大模型](https://zhuanlan.zhihu.com/p/656320578)
3. [如何评估大模型指令跟随能力](https://blog.csdn.net/2401_85549225/article/details/139808547)
4. 

## 大模型应用与大模型agent

### 大模型应用

1. [langchain中文文档](https://js.langchain.com.cn/docs/)
2. [huggingface TGI](https://blog.csdn.net/hyang1974/article/details/138501231)
3. [大模型和推荐有什么结合方式？](https://blog.csdn.net/Thanours/article/details/139319316)
4. 一些code gpt
5. 大模型生成数据、数据清洗策略
6. 大模型记忆
7. [部分应用总结](https://mp.weixin.qq.com/s/1uOtz340I1UBxitcwTzxtA)
8. 大模型+弱监督学习的零样本相关性模型
基于弱监督学习的零样本标注方案


### 大模型智能体
1. magnus

## 其他问题

### git版本控制

1. [git merge和git rebase的区别？](https://blog.csdn.net/weixin_45565886/article/details/133798840)
2. [git 冲突如何解决](https://blog.csdn.net/weixin_44799217/article/details/132013096)

### 其他计算机知识
1. [流水线并行训练](https://zhuanlan.zhihu.com/p/707784244)
2. [minhash原理](https://blog.csdn.net/zfhsfdhdfajhsr/article/details/128529402)
3. [all reduce](https://blog.csdn.net/qq_38342510/article/details/136359296)
4. [容灾](https://baijiahao.baidu.com/s?id=1673450846247020471&wfr=spider&for=pc)
5. docker基础
6. code review

### python基础
1. [python的迭代器和生成器的区别](https://blog.csdn.net/qq_52758588/article/details/136643799)
2. [装饰器](https://www.runoob.com/python3/python-decorators.html)
3. [*args、**kwargs的使用](https://blog.csdn.net/sodaloveer/article/details/134165294)

### 搜推相关
ab实验，holdout ，gsb，评测指标gmv,ucvr

## pytorch基础
- 注：可能会在面试过程中考察手撕代码，包括但不限于基本训练代码（MLP+梯度更新）、normalization、经典模型的forward、损失函数、多头注意力机制
1. [torchrun命令的使用](https://blog.51cto.com/u_15887260/7733758)


## 贡献指南

欢迎大家通过提交Issue或Pull Request来贡献题目和答案。在贡献之前，请确保您遵循以下指南：

1. 确保提交的题目和答案清晰明了，并且经过验证。
2. 在提交Pull Request之前，请检查是否有重复的题目。
3. 请遵循Markdown格式，并保持项目的一致性。
