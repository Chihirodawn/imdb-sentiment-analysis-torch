# IMDB 影评情感分类：GloVe 与深度学习模型对比

本项目完成 IMDB 影评二分类任务，比较了基于 GloVe 词向量的经典深度学习网络，以及 BERT、DistilBERT、RoBERTa 等预训练 Transformer。全部模型均生成了测试集预测，并在 Kaggle 的 **Bag of Words Meets Bags of Popcorn** 评测中获得成绩。

## 模型与结果

| Kaggle 排名 | 模型 | 方法类别 | 本地最佳验证准确率 | Kaggle Score |
|---:|---|---|---:|---:|
| 1 | RoBERTa | 预训练 Transformer | 94.50% | **95.112%** |
| 2 | BERT | 预训练 Transformer | 93.40% | **93.852%** |
| 3 | DistilBERT | 预训练 Transformer | 93.18% | **93.128%** |
| 4 | GRU | GloVe + RNN | 90.57% | **90.088%** |
| 5 | LSTM | GloVe + RNN | 89.73% | **89.380%** |
| 6 | CNN | GloVe + CNN | 90.23% | **89.020%** |
| 7 | Attention-LSTM | GloVe + Attention | 86.00% | **85.660%** |
| 8 | Capsule-LSTM | GloVe + Capsule | 86.00% | **84.740%** |
| 9 | CNN-LSTM | GloVe + CNN/RNN | 78.00% | **77.068%** |
| 10 | Transformer | GloVe + Transformer Encoder | 76.00% | **70.440%** |

Kaggle Public Score 与 Private Score 显示一致。最佳模型为 **RoBERTa**，Kaggle 准确率为 **95.112%**。

> Transformer 的日志最佳验证准确率出现在第 8 轮；当前保存的提交结果来自最终轮，因此其 Kaggle 分数低于日志中的最高验证结果。

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
│   ├── imdb_bert_trainer.py      # BERT
│   ├── imdb_distilbert_trainer.py# DistilBERT
│   └── imdb_roberta_trainer.py   # RoBERTa
├── tools/convert_glove.py        # GloVe / Gensim 格式转换工具
├── results/                      # 10 个模型的 Kaggle 预测 CSV
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
```

运行后会在 `result/` 生成格式为 `id,sentiment` 的预测文件；本仓库中已有的对应预测文件保存在 `results/`，可用于核对实验结果。

## 结论

1. 预训练语言模型整体显著优于仅使用 GloVe 的经典网络；RoBERTa 得到最高 Kaggle Score（95.112%）。
2. 在 GloVe 经典模型中，GRU 的 Kaggle 表现最佳（90.088%）；LSTM 与 CNN 紧随其后。
3. Attention-LSTM 与 Capsule-LSTM 相比普通 RNN 未取得更高分数，说明该任务中预训练模型和表示能力的影响更显著。
4. 结果 CSV、源代码与依赖清单均已保留，便于复现和对比。
