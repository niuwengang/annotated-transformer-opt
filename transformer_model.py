"""
Transformer 模型架构组件
========================
基于论文 "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
来源于 HarvardNLP annotated-transformer 项目
"""

import math
import copy
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


# ============================================================================
# 基础工具
# ============================================================================

def clones(module, N):
    """复制 N 个相同的层"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    残差连接 + 层归一化
    为了代码简洁，将层归一化放在残差连接之前
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  #SublayerConnection 必须是一个能接收 x 的函数
        return x + self.dropout(sublayer(self.norm(x)))  #Pre-Norm(论文里是Post-Norm，代码这个效果更好)


# ============================================================================
# Attention 机制
# ============================================================================

def attention(query, key, value, mask=None, dropout=None):
    """计算缩放点积注意力 (Scaled Dot-Product Attention)"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    """屏蔽后续位置的注意力（上三角矩阵）"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class MultiHeadedAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()  #调用父类 nn.Module 的初始化
        assert d_model % h == 0  # d_model 必须是 h 的整数倍
        self.d_k = d_model // h #每个头的维度处理64维度
        self.h = h  #8个头
        self.linears = clones(nn.Linear(d_model, d_model), 4)# 4 个 512×512 线性层 QKV和O
        self.attn = None  #存储注意力权重，便于可视化
        self.dropout = nn.Dropout(p=dropout)  #随机丢弃 10% 神经元，防过拟合

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) #给 mask 加一个维度：[batch, seq, seq] → [batch, 1, seq, seq]
        nbatches = query.size(0) #获取 batch 大小，比如一次训练 64 句话

        # 1) 线性投影：d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 拼接多头结果，再过最后一个线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query, key, value
        return self.linears[-1](x)


# ============================================================================
# 前馈网络
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """两层全连接前馈网络 FFN(x) = max(0, xW1+b1)W2+b2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) #线性层
        self.w_2 = nn.Linear(d_ff, d_model) ##线性层
        self.dropout = nn.Dropout(dropout) #随机失活

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# ============================================================================
# Embeddings 和 位置编码
# ============================================================================

class Embeddings(nn.Module):
    """词嵌入层，乘以 sqrt(d_model)"""

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码"""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ============================================================================
# Encoder 组件
# ============================================================================

class EncoderLayer(nn.Module):
    """编码器层：自注意力 + 前馈网络"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn   #自注意力
        self.feed_forward = feed_forward  #前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  #向量维度

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #自注意力
        return self.sublayer[1](x, self.feed_forward) #FFN 非线性变换 升维


class Encoder(nn.Module):
    """N 层编码器堆叠"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  #复制 N 个编码器层
        self.norm = LayerNorm(layer.size) #层归一化

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ============================================================================
# Decoder 组件
# ============================================================================

class DecoderLayer(nn.Module):
    """解码器层：自注意力 + 源注意力 + 前馈网络"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size #向量维度
        self.self_attn = self_attn #自注意力
        self.src_attn = src_attn #源注意力
        self.feed_forward = feed_forward #前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #掩码自注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) #交叉注意力 Q匹配 K被匹配 V内容输入
        return self.sublayer[2](x, self.feed_forward) #前馈网络


class Decoder(nn.Module):
    """N 层解码器堆叠"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================================
# 生成器
# ============================================================================

class Generator(nn.Module):
    """线性 + log_softmax 生成步骤"""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


# ============================================================================
# 完整 Encoder-Decoder 模型
# ============================================================================

class EncoderDecoder(nn.Module):
    """标准的编码器-解码器架构"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder #编码器
        self.decoder = decoder #解码器
        self.src_embed = src_embed #词嵌入层
        self.tgt_embed = tgt_embed #词嵌入层
        self.generator = generator #生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# ============================================================================
# 模型构建函数
# ============================================================================

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """根据超参数构建完整 Transformer 模型"""
    c = copy.deepcopy #深拷贝
    attn = MultiHeadedAttention(h, d_model) #多头注意力机制模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络模块
    position = PositionalEncoding(d_model, dropout)  #位置编码模块
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),#源嵌入
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),#目标嵌入
        Generator(d_model, tgt_vocab), #log_softmax 层
    )

    # Xavier 初始化 (让每一层输入的方差等于输出的方差)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ============================================================================
# 贪婪解码
# ============================================================================

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """使用贪婪解码生成翻译结果"""
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys 