"""Counter 用法演示 —— 与 transformer_data.py 中 build_vocab_from_iterator 的逻辑一致"""

from collections import Counter


def main():
    # ============================================================
    # 1. 模拟一个 iterator：每次 yield 一句话（token 列表）
    # ============================================================
    def mock_iterator():
        yield ["I", "love", "NLP", "love"]
        yield ["NLP", "is", "fun"]
        yield ["I", "love", "transformer"]

    # ============================================================
    # 2. 用 Counter 统计词频（和 build_vocab_from_iterator 一样）
    # ============================================================
    counter = Counter()
    for tokens in mock_iterator():
        counter.update(tokens)

    print("词频统计结果：")
    print(counter)
    print()

    # ============================================================
    # 3. 常用操作
    # ============================================================
    # 出现次数最多的 2 个词
    print("most_common(2):", counter.most_common(2))

    # 不存在的 key 返回 0，不会报错
    print("counter['unknown'] =", counter["unknown"])

    # 总词数
    print("总词数:", sum(counter.values()))  # total() 需要 Python 3.10+，用 sum 兼容
    print()

    # ============================================================
    # 4. 模拟 SimpleVocab 的过滤逻辑：min_freq 过滤低频词
    # ============================================================
    min_freq = 2
    print(f"词频统计（过滤 min_freq={min_freq}）：")
    for word, freq in counter.most_common():
        if freq >= min_freq:
            print(f"  {word}: {freq}  ✓ 保留")
        else:
            print(f"  {word}: {freq}  ✗ 丢弃")


if __name__ == "__main__":
    main()