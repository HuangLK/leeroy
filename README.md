
## Leeroy
一个基于pytorch-lightning和huggingface的NLP模型训练框架。pytorch-lightning trainer能极大简化训练的代码，huggingface提供了几乎完备的transformer模型库，结合两者实现常见的任务（如分类、序列标注、MRC等），能节省大量训练流程和模型设计的编码时间，能有更多的时间关注业务和数据。

## 功能点
* 支持分类任务

## fixed bugs
* 修复val结束之后metrics未reset

## TODO
* 支持多标签多分类任务
* 增加checkpoint恢复训练（集群抢占资源必备）
* 增加其他任务（持续工作）

## 协作共建
建议采用[分支管理](http://www.ruanyifeng.com/blog/2012/07/git.html)。
