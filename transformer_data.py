"""
数据加载、词表构建、分词
========================
自定义 Multi30k 数据集加载器，不依赖 torchtext
"""

import os
import gzip
import urllib.request
from collections import Counter
from os.path import exists

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import spacy


# ============================================================================
# 自定义词表 (替代 torchtext.vocab)
# ============================================================================

class SimpleVocab:
    """自定义词表（替代 torchtext.vocab.build_vocab_from_iterator）"""

    def __init__(self, counter, specials, min_freq=1):
        self.itos = specials.copy()
        self.stoi = {token: i for i, token in enumerate(specials)}
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        self.default_index = 0

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def set_default_index(self, idx):
        self.default_index = idx


def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    """从迭代器构建词表（替代 torchtext.vocab.build_vocab_from_iterator）"""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    if specials is None:
        specials = []
    return SimpleVocab(counter, specials, min_freq=min_freq)


def to_map_style_dataset(iterable_dataset):
    """将可迭代数据集转为 map-style 数据集"""
    return list(iterable_dataset)


# ============================================================================
# Multi30k 数据集加载器
# ============================================================================

class Multi30kDataset:
    """自定义 Multi30k 数据集加载器，不依赖 torchtext"""

    _URLS = {
        "train": {
            "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz",
            "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz",
        },
        "val": {
            "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz",
            "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz",
        },
        "test_2016": {
            "de": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz",
            "en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz",
        },
    }

    def __init__(self, language_pair=("de", "en"), root=".data"):
        self.language_pair = language_pair
        self.root = root
        self._data = None

    def _download_and_read(self, split):
        os.makedirs(self.root, exist_ok=True)
        lines = {}
        for lang in self.language_pair:
            url = self._URLS[split][lang]
            filename = os.path.join(self.root, f"multi30k_{split}.{lang}")
            if not os.path.exists(filename):
                print(f"Downloading {url} ...")
                tmpfile = filename + ".gz.tmp"
                urllib.request.urlretrieve(url, tmpfile)
                with gzip.open(tmpfile, "rt", encoding="utf-8") as f_in:
                    data = f_in.read()
                with open(filename, "w", encoding="utf-8") as f_out:
                    f_out.write(data)
                os.remove(tmpfile)
            with open(filename, "r", encoding="utf-8") as f:
                lines[lang] = [line.strip() for line in f.readlines()]
        return list(zip(lines[self.language_pair[0]], lines[self.language_pair[1]]))

    def __iter__(self):
        """迭代模式：返回验证集数据"""
        return iter(self._download_and_read("val"))

    def __call__(self, language_pair=None):
        """返回 train, val, test 三个数据集"""
        if language_pair is not None:
            self.language_pair = language_pair
        train_data = self._download_and_read("train")
        val_data = self._download_and_read("val")
        test_data = self._download_and_read("test_2016")
        return train_data, val_data, test_data


def load_multi30k(language_pair=("de", "en")):
    """加载 Multi30k 数据集，返回 train, val, test"""
    dataset = Multi30kDataset(language_pair=language_pair)
    return dataset()


# ============================================================================
# 分词 (Spacy)
# ============================================================================

def load_tokenizers():
    """加载 Spacy 分词器"""
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    """使用 Spacy 分词器对文本进行分词"""
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    """从数据迭代器中逐条产出分词结果"""
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# ============================================================================
# 词表构建与加载
# ============================================================================

def build_vocabulary(spacy_de, spacy_en):
    """构建德语和英语词表"""
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = load_multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = load_multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en, vocab_path="vocab.pt"):
    """加载或构建词表"""
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path)
    print(f"Vocabulary sizes: src={len(vocab_src)}, tgt={len(vocab_tgt)}")
    return vocab_src, vocab_tgt


# ============================================================================
# DataLoader 构建
# ============================================================================

def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab,
                  device, max_padding=128, pad_id=2):
    """批次整理函数：分词、转ID、填充"""
    bs_id = torch.tensor([0], device=device)   # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []

    for (_src, _tgt) in batch:
        processed_src = torch.cat([
            bs_id,
            torch.tensor(src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device),
            eos_id,
        ], 0)
        processed_tgt = torch.cat([
            bs_id,
            torch.tensor(tgt_vocab(tgt_pipeline(_tgt)), dtype=torch.int64, device=device),
            eos_id,
        ], 0)

        src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id))
        tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en,
                       batch_size=12000, max_padding=128, is_distributed=True):
    """创建训练和验证 DataLoader"""
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(batch, tokenize_de, tokenize_en,
                             vocab_src, vocab_tgt, device,
                             max_padding=max_padding,
                             pad_id=vocab_src.get_stoi()["<blank>"])

    train_iter, valid_iter, test_iter = load_multi30k(language_pair=("de", "en"))

    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map, batch_size=batch_size,
        shuffle=(valid_sampler is None), sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader