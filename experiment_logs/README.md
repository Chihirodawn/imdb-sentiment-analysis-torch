# 实验日志说明

本目录保留训练过程的原始日志，用于核验实验、复现模型执行顺序及排查环境问题。

- `original_2026_07_11/`：首次完整实验。含数据预处理、GloVe 下载/转换、10 个正式 Kaggle 提交模型的训练日志、重跑队列日志和 TensorBoard 事件文件。
- `bert_rerun_2026_07_13/`：RTX 4090 上的 BERT 系列复训。含四个训练日志、两份调度日志及 `bert_retraining_status.tsv`。该状态表确认四个脚本全部成功完成。

`imdb_bert_native_initial_failed.log` 记录了 Hugging Face 官方站点连接超时的首次失败，不代表实验结果；成功结果应以同目录的 `imdb_bert_native_rerun.log` 为准。

日志包含逐 batch 进度，因此不在项目首页全文展示；关键准确率和异常处理结论已汇总在仓库根目录的 `README.md`。
