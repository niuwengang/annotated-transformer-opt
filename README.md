# 快速测试：简单复制任务（合成数据，20 epoch，无需下载数据集）
python main.py --mode simple

# 训练 Multi30k 德→英翻译模型（8 epoch，默认参数）
python main.py --mode train

# 自定义参数训练
python main.py --mode train --epochs 5 --batch_size 64

# 多 GPU 分布式训练
python main.py --mode train --distributed

# 评估已训练模型，展示翻译结果
python main.py --mode eval

# 评估指定条数
python main.py --mode eval --n_examples 5