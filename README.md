# ChatGLM2-6B-Explained

ChatGLM2-6B-相关代码，逐行详解版。  
逐步更新，欢迎大家Star，Fork，参与进来，提交PR。   
注：xxx表示伪目录，非有效。

##
这个项目主要是数据相关的流转，测试，还有p tuning v2相关微调。若是想弄懂大模型的原理，建议看[GLM-Explained](https://github.com/ArtificialZeng/GLM-Explained)

此外，大模型还基于两个非常重要的基础库，那便是[transformers](https://github.com/ArtificialZeng/tranformers-expalined)，和[pytorch](https://github.com/ArtificialZeng/pytorch-explained)，同样这两个库也有关键代码的逐行解析版本。
# ChatGLM2-6B-Explained



* [x/](./src)
  * [x/](./src/utils)
    * [main.py](./ptuning/main.py)
    * [train.sh参数解释](./ptuning/train.sh) 
  * [x.py](./src/train_sft.py)
* [/configuration_chatglm.py](./chatglm2PT/configuration_chatglm.py)  这段代码定义了一个名为ChatGLMConfig的类，用于配置和管理ChatGLM模型。
* 
* [x/](./examples)
  * [x.md](./examples/ads_generation.md)
* [README.md](./README.md)


# CSDN彩色博客版：
* [ChatGLM1/2 系列源码解析系列-专栏地址](https://blog.csdn.net/sinat_37574187/category_12365053.html) 
  * [/src/utils/](./ChatGLM-Efficient-Tuning-Explained/src/utils)
    * [CSDN彩色源码解析main.py(一)](https://zengxiaojian.blog.csdn.net/article/details/131617133?spm=1001.2014.3001.5502)
    * [CSDN彩色源码解析main.py(二)](https://blog.csdn.net/sinat_37574187/article/details/131621397)
* [ChatGLM2-6B源码解析 web_demo.py](https://blog.csdn.net/sinat_37574187/article/details/131404024)
* [README.md](./ChatGLM-Efficient-Tuning-Explained/README.md)


## 引用 - 源项目
