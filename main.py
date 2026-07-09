#!/usr/bin/env python3
"""
The Annotated Transformer — 传统 main() 入口程序
===================================================
基于论文 "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
来源于 HarvardNLP annotated-transformer 项目

用法：
    python main.py --mode simple          # 运行简单复制任务（快速测试）
    python main.py --mode train           # 训练 Multi30k 德→英翻译模型
    python main.py --mode eval            # 评估已训练模型，展示翻译结果
    python main.py --mode eval --epochs 10 --batch_size 64  # 自定义参数训练
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="The Annotated Transformer — 训练与评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode simple                     # 简单复制任务
  python main.py --mode train                      # 训练翻译模型
  python main.py --mode train --epochs 5 --batch_size 64
  python main.py --mode eval                       # 评估翻译结果
  python main.py --mode eval --n_examples 5        # 评估 5 条翻译
        """,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["simple", "train", "eval"],
        help="运行模式: simple=简单复制任务, train=训练翻译模型, eval=评估",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=8,
                        help="训练轮数 (默认: 8)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--base_lr", type=float, default=1.0,
                        help="基础学习率 (默认: 1.0)")
    parser.add_argument("--warmup", type=int, default=3000,
                        help="Warmup 步数 (默认: 3000)")
    parser.add_argument("--accum_iter", type=int, default=10,
                        help="梯度累积步数 (默认: 10)")
    parser.add_argument("--max_padding", type=int, default=72,
                        help="最大填充长度 (默认: 72)")
    parser.add_argument("--model_prefix", type=str, default="multi30k_model_",
                        help="模型保存前缀 (默认: multi30k_model_)")
    parser.add_argument("--model_path", type=str, default="multi30k_model_final.pt",
                        help="要加载的模型路径 (默认: multi30k_model_final.pt)")
    parser.add_argument("--vocab_path", type=str, default="vocab.pt",
                        help="词表路径 (默认: vocab.pt)")
    parser.add_argument("--n_examples", type=int, default=10,
                        help="评估时展示的翻译样例数 (默认: 10)")

    # 分布式训练
    parser.add_argument("--distributed", action="store_true",
                        help="启用多 GPU 分布式训练")

    args = parser.parse_args()

    # ── 模式: simple ──────────────────────────────────────────
    if args.mode == "simple":
        from transformer_training import run_simple_copy_task
        run_simple_copy_task()
        return

    # ── 模式: train / eval（需要加载数据）──────────────────────
    print("Loading tokenizers ...")
    from transformer_data import load_tokenizers, load_vocab
    spacy_de, spacy_en = load_tokenizers()  #获取分词器

    print("Loading vocabulary ...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, vocab_path=args.vocab_path) #获取英语和德语的词表

    if args.mode == "train":
        config = {
            "batch_size": args.batch_size, # 批次大小
            "distributed": args.distributed, # 是否启用分布式训练
            "num_epochs": args.epochs, # 训练轮数
            "accum_iter": args.accum_iter, # 每 accum_iter 个 batch 才执行一次 optimizer.step() 更新参数
            "base_lr": args.base_lr,# 基础学习率，后续由rate() 调度函数动态计算
            "max_padding": args.max_padding,# 最大填充长度，一个 batch 内所有句子统一填充到的最大 token 长度。
            "warmup": args.warmup,# warmup 步数
            "file_prefix": args.model_prefix, # 模型保存前缀
        }
        print("Training config:", config)
        from transformer_training import train_model
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
        print("Training finished!")

    elif args.mode == "eval":
        from transformer_training import run_evaluation
        run_evaluation(
            vocab_src, vocab_tgt, spacy_de, spacy_en,
            model_path=args.model_path,
            n_examples=args.n_examples,
        )


if __name__ == "__main__":
    main()