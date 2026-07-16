# 实验日志说明

本目录保留成功训练过程的日志，用于核验实验结果和复现模型执行流程。

- `original_2026_07_11/`：数据预处理、GloVe 下载与转换、经典深度学习模型、Transformer、BERT、DistilBERT、RoBERTa 的训练日志和 TensorBoard 事件文件。
- `bert_rerun_2026_07_13/`：BERT Native、BERT Scratch、BERT Trainer、DistilBERT Native 的成功训练日志、调度日志及状态表。
- `transformer_scale_2026_07_16/`：BERT、RoBERTa、DeBERTa 的 base/large 对比实验日志，以及六模型顺序执行状态记录。

日志包含逐 batch 训练进度，因此不在项目首页全文展示；关键准确率、Kaggle 排名和实验结论已汇总在仓库根目录的 `README.md`。
