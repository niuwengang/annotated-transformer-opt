# Annotated Transformer 优化

基于论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 的 Transformer 实现，来源于 [HarvardNLP annotated-transformer](https://github.com/harvardnlp/annotated-transformer) 项目。本项目对原始代码进行了拆分重构，将 Jupyter Notebook 拆分为独立的 Python 模块，并添加了命令行入口，方便训练与评估。

## 项目结构

```
.
├── main.py                  # 命令行入口（训练 / 评估 / 快速测试）
├── transformer_model.py     # 模型架构：Attention、Encoder、Decoder、位置编码等
├── transformer_training.py  # 训练工具：Batch、TrainState、学习率调度、标签平滑、分布式训练
├── transformer_data.py      # 数据加载：Multi30k 数据集、Spacy 分词、词表构建
├── requirements.txt         # 依赖列表
└── README.md
```

## 模型架构

- **Encoder**: 6 层，每层包含多头自注意力（8 heads） + 前馈网络（d_ff=2048），d_model=512
- **Decoder**: 6 层，每层包含掩码自注意力 + 交叉注意力 + 前馈网络
- **位置编码**: 正弦/余弦位置编码
- **正则化**: 标签平滑（smoothing=0.1）、Dropout（0.1）
- **优化器**: Adam（β₁=0.9, β₂=0.98, ε=10⁻⁹）
- **学习率调度**: Warmup 线性增长 + 平方根衰减（warmup=3000 steps）

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：PyTorch 1.11、torchtext 0.12、spaCy 3.2（德语/英语分词模型）、GPUtil、wandb 等。

## 运行

### 快速测试：简单复制任务

```bash
python main.py --mode simple
```

用合成数据训练一个小型 Transformer（N=2，随机数字序列复制），快速验证模型和训练流程是否正常。

### 训练 Multi30k 德→英翻译模型

```bash
python main.py --mode train
```

默认参数：8 epoch，batch_size=32，梯度累积 10 步，每条数据最多 72 token。

### 自定义参数训练

```bash
python main.py --mode train --epochs 5 --batch_size 64
```

### 多 GPU 分布式训练

```bash
python main.py --mode train --distributed
```

基于 PyTorch DistributedDataParallel（NCCL 后端），自动检测可用 GPU 数量。

### 评估已训练模型

```bash
python main.py --mode eval
```

加载训练好的模型，在验证集上展示翻译结果（默认 10 条）。

### 评估指定条数

```bash
python main.py --mode eval --n_examples 5
```

## 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | 必选 | `simple` / `train` / `eval` |
| `--epochs` | int | 8 | 训练轮数 |
| `--batch_size` | int | 32 | 批次大小 |
| `--base_lr` | float | 1.0 | 基础学习率 |
| `--warmup` | int | 3000 | Warmup 步数 |
| `--accum_iter` | int | 10 | 梯度累积步数 |
| `--max_padding` | int | 72 | 最大填充长度 |
| `--model_prefix` | str | multi30k_model_ | 模型保存前缀 |
| `--model_path` | str | multi30k_model_final.pt | 评估时加载的模型路径 |
| `--vocab_path` | str | vocab.pt | 词表文件路径 |
| `--n_examples` | int | 10 | 评估展示翻译样例数 |
| `--distributed` | flag | False | 启用多 GPU 分布式训练 |

## 数据集

使用 [Multi30k](https://github.com/multi30k/dataset) 数据集（德→英翻译），包含约 30,000 条训练数据。首次运行时会自动从 GitHub 下载并缓存到 `.data/` 目录。

## 参考

- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)]
- [HarvardNLP annotated-transformer](https://github.com/harvardnlp/annotated-transformer) — 原始 Jupyter Notebook 实现
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — 在线注释版

## License

MIT License