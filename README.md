
## Leeroy
一个基于pytorch-lightning和🤗transformers的NLP模型训练框架。pytorch-lightning trainer能极大简化训练的代码，🤗transformers提供了几乎完备的transformer模型库，结合两者实现常见的任务（如分类、序列标注、MRC等），能节省大量训练流程和模型设计的编码时间，能有更多的时间关注业务和数据。

## features
* 支持多分类任务
* 支持多标签分类任务
* 支持UIE式的抽取任务，能直接加载UIE参数
* 支持seq2seq生成任务

## fixed bugs
*

## TODO
* 增加checkpoint恢复训练（集群抢占资源必备）
* 增加loss配置机制，支持Focal loss等，方便炼丹
* 增加其他任务（持续工作）

## 协作共建
建议采用[分支管理](http://www.ruanyifeng.com/blog/2012/07/git.html)。
