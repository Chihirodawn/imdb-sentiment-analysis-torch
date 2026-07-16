# IMDB 影评情感分类：GloVe 与深度学习模型对比

本项目完成 IMDB 影评二分类任务，比较基于 GloVe 词向量的经典深度学习网络，以及 BERT、DistilBERT、RoBERTa、DeBERTa 等预训练 Transformer。所有模型均完成训练与测试集预测；原有模型已提交到 Kaggle 的 **Bag of Words Meets Bags of Popcorn** 竞赛进行评测，并补充了三组 Transformer 的 base/large 对比实验。

## 模型与结果

| Kaggle 排名 | 模型 | 方法类别 | 本地最佳验证准确率 | Kaggle Score |
|---:|---|---|---:|---:|
| 1 | DeBERTa-v3-large | 预训练 Transformer（large） | 96.72% | **96.688%** |
| 2 | RoBERTa-large | 预训练 Transformer（large） | 96.36% | **96.412%** |
| 3 | DeBERTa-v3-base | 预训练 Transformer（base） | 95.20% | **95.496%** |
| 4 | RoBERTa | 预训练 Transformer | 94.50% | **95.112%** |
| 5 | RoBERTa-base | 预训练 Transformer（base） | 94.60% | **94.704%** |
| 6 | BERT Trainer | 预训练 Transformer（Trainer） | 93.38% | **94.072%** |
| 7 | BERT-large | 预训练 Transformer（large） | 93.86% | **94.056%** |
| 8 | BERT Scratch | 预训练 Transformer + 自定义分类模型 | 92.86% | **93.424%** |
| 9 | DistilBERT Trainer | 轻量化预训练 Transformer（Trainer） | 93.18% | **93.128%** |
| 10 | DistilBERT Native | 轻量化预训练 Transformer（原生 PyTorch） | 92.00% | **91.376%** |
| 11 | BERT Native | 预训练 Transformer（原生 PyTorch） | 91.00% | **90.464%** |
| 12 | GRU | GloVe + RNN | 90.57% | **90.088%** |
| 13 | LSTM | GloVe + RNN | 89.73% | **89.380%** |
| 14 | CNN | GloVe + CNN | 90.23% | **89.020%** |
| 15 | Attention-LSTM | GloVe + Attention | 86.00% | **85.660%** |
| 16 | Capsule-LSTM | GloVe + Capsule | 86.00% | **84.740%** |
| 17 | CNN-LSTM | GloVe + CNN/RNN | 78.00% | **77.068%** |
| 18 | Transformer | GloVe + Transformer Encoder | 76.00% | **70.440%** |

Kaggle Public Score 与 Private Score 显示一致。当前最佳模型为 **DeBERTa-v3-large**，Kaggle 准确率为 **96.688%**。BERT-base 的预测文件已经保留，但本次截图未显示其 Kaggle Score，因此暂未加入本榜单。

> Transformer 的日志最佳验证准确率出现在第 8 轮；当前保存的提交结果来自最终轮，因此其 Kaggle 分数低于日志中的最高验证结果。

## Transformer base/large 验证集排行（2026-07-16）

| 本地排名 | 模型 | 最佳验证准确率 | Kaggle Score | 最佳 Epoch | 关键训练设置 | 预测文件 |
|---:|---|---:|---:|---:|---|---|
| 1 | DeBERTa-v3-large | **96.72%** | **96.688%** | 3 | LR `1e-5`，Batch 4，梯度累积 4 | `results/deberta_v3_large_trainer.csv` |
| 2 | RoBERTa-large | **96.36%** | **96.412%** | 2 | LR `1e-5`，Batch 16 | `results/roberta_large_trainer.csv` |
| 3 | DeBERTa-v3-base | **95.20%** | **95.496%** | 3 | 默认 LR `5e-5`，Batch 16 | `results/deberta_v3_base_trainer.csv` |
| 4 | RoBERTa-base | **94.60%** | **94.704%** | 3 | 默认 LR `5e-5`，Batch 16 | `results/roberta_base_trainer.csv` |
| 5 | BERT-large | **93.86%** | **94.056%** | 3 | 默认 LR `5e-5`，Batch 16 | `results/bert_large_trainer.csv` |
| 6 | BERT-base | **93.50%** | 待补 | 3 | 默认 LR `5e-5`，Batch 16 | `results/bert_base_trainer.csv` |

以上六组实验使用相同的数据划分（随机种子 42）、`max_length=512` 和 3 个 Epoch。BERT、RoBERTa、DeBERTa 的 large 相比对应 base 分别提升 0.36、1.76、1.52 个百分点。RoBERTa-large 和 DeBERTa-large 在默认学习率 `5e-5` 下未正常收敛，因此改用 `1e-5`；DeBERTa-large 在 RTX 5090 32GB 上以 Batch 16 训练会显存溢出，因此使用 Batch 4、梯度累积 4，保持等效 Batch 约为 16。

> 本节排名依据本地验证集最佳准确率，不是 Kaggle Score；六个测试集预测 CSV 已保存在 `results/`，可继续提交 Kaggle 评测。

## 项目结构

```text
.
├── src/                         # 训练和预测脚本
│   ├── imdb_process.py           # 数据清洗、划分、GloVe 特征构建
│   ├── imdb_cnn.py               # CNN
│   ├── imdb_lstm.py              # LSTM
│   ├── imdb_gru.py               # GRU
│   ├── imdb_attention_lstm.py    # Attention-LSTM
│   ├── imdb_capsule_lstm.py      # Capsule-LSTM
│   ├── imdb_cnnlstm.py           # CNN-LSTM
│   ├── imdb_transformer.py       # Transformer Encoder
│   ├── imdb_bert_native.py       # BERT（原生 PyTorch）
│   ├── imdb_bert_scratch.py      # BERT（自定义分类模型）
│   ├── imdb_bert_trainer.py      # BERT（Hugging Face Trainer）
│   ├── imdb_distilbert_native.py # DistilBERT（原生 PyTorch）
│   ├── imdb_distilbert_trainer.py # DistilBERT（Trainer）
│   ├── imdb_roberta_trainer.py   # 原有 RoBERTa
│   ├── imdb_bert_base_trainer.py # BERT-base
│   ├── imdb_bert_large_trainer.py # BERT-large
│   ├── imdb_roberta_base_trainer.py # RoBERTa-base
│   ├── imdb_roberta_large_trainer.py # RoBERTa-large
│   ├── imdb_deberta_base_trainer.py # DeBERTa-v3-base
│   └── imdb_deberta_large_trainer.py # DeBERTa-v3-large
├── tools/convert_glove.py        # GloVe / Gensim 格式转换工具
├── results/                      # 所有模型的 Kaggle 提交 CSV
├── experiment_logs/              # 成功训练、依赖与调度日志
│   ├── original_2026_07_11/      # 首次完整实验日志
│   └── bert_rerun_2026_07_13/    # BERT 系列训练日志与状态表
└── requirements.txt
```

## 数据与环境

- Python 3.10+、PyTorch、CUDA（或 Apple Silicon MPS）
- IMDB 数据：Kaggle `word2vec-nlp-tutorial` 的 `labeledTrainData.tsv`、`testData.tsv`、`unlabeledTrainData.tsv`
- 词向量：Stanford GloVe `glove.840B.300d.txt`

为避免仓库体积过大，IMDB 原始数据、GloVe 向量和训练权重均不包含在仓库中。将数据按如下路径放置：

```text
corpus/imdb/labeledTrainData.tsv
corpus/imdb/testData.tsv
corpus/imdb/unlabeledTrainData.tsv
glove/glove.840B.300d.txt
```

安装依赖：

```bash
pip install -r requirements.txt
# 再按本机 CUDA / MPS 环境安装对应版本的 PyTorch
```

## 复现流程

先从仓库根目录运行数据预处理：

```bash
python src/imdb_process.py
```

该步骤会生成 `pickle/imdb_glove.pickle3`。之后可运行任意模型脚本，例如：

```bash
mkdir -p result
python src/imdb_cnn.py
python src/imdb_lstm.py
python src/imdb_gru.py
python src/imdb_attention_lstm.py
python src/imdb_roberta_trainer.py
python src/imdb_roberta_large_trainer.py
python src/imdb_deberta_large_trainer.py
```

运行后会在 `result/` 生成格式为 `id,sentiment` 的预测文件；本仓库中已有的对应预测文件保存在 `results/`，可用于核对实验结果。

## 日志与过程记录

成功训练日志均已保存在 `experiment_logs/`，包括数据预处理、GloVe 下载与转换、经典深度学习模型、Transformer、BERT、DistilBERT、RoBERTa，以及 TensorBoard 事件文件。README 汇总最终指标和实验结论；每个模型的详细训练过程可在对应 `.log` 文件中核验。

## 结论

1. 预训练语言模型整体显著优于仅使用 GloVe 的经典网络；新增的 DeBERTa-v3-large 得到最高 Kaggle Score（96.688%），RoBERTa-large（96.412%）和 DeBERTa-v3-base（95.496%）分列第二、第三。
2. BERT 与 DistilBERT 的 Trainer 实现均优于对应的原生 PyTorch 实现，说明训练策略和实现方式会明显影响最终效果。
3. 在 GloVe 经典模型中，GRU 的 Kaggle 表现最佳（90.088%）；LSTM 与 CNN 紧随其后。
4. Attention-LSTM 与 Capsule-LSTM 相比普通 RNN 未取得更高分数，说明该任务中预训练模型和表示能力的影响更显著。
5. 所有提交 CSV、源代码、训练日志与依赖清单均已保留，便于复现、核验和对比。
6. 在相同数据划分下，三组 large 模型均优于对应 base；DeBERTa-v3-large 的本地最佳验证准确率为 96.72%，Kaggle Score 为 96.688%，但 large 模型对显存和学习率也更敏感。
