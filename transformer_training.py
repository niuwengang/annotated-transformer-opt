"""
训练工具
========
包含 Batch、TrainState、run_epoch、学习率调度、标签平滑损失、
以及完整的训练 worker 函数
"""

import time
import os
from os.path import exists

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import GPUtil

# 从模型和数据模块导入所需
from transformer_model import make_model, greedy_decode, subsequent_mask
from transformer_data import create_dataloaders


# ============================================================================
# 哑优化器和调度器（仅用于评估模式）
# ============================================================================

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass


# ============================================================================
# 批次数据
# ============================================================================

class Batch:
    """用于在训练过程中保存一个数据批次及其掩码"""

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """创建掩码：隐藏填充位置和未来位置"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


# ============================================================================
# 训练状态追踪
# ============================================================================

class TrainState:
    """跟踪步数、样本数量和处理的标记数量"""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


# ============================================================================
# 训练循环
# ============================================================================

def run_epoch(data_iter, model, loss_compute, optimizer, scheduler,
              mode="train", accum_iter=1, train_state=None):
    """训练一个 epoch"""
    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                ("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                 + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e")
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

        del loss, loss_node

    return total_loss / total_tokens, train_state


# ============================================================================
# 学习率调度
# ============================================================================

def rate(step, model_size, factor, warmup):
    """学习率调度函数：warmup 阶段线性增长，之后按平方根衰减"""
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# ============================================================================
# 标签平滑损失
# ============================================================================

class LabelSmoothing(nn.Module):
    """标签平滑正则化"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# ============================================================================
# 简单损失计算
# ============================================================================

class SimpleLossCompute:
    """简单的损失计算和训练函数"""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),
                y.contiguous().view(-1)
            ) / norm
        )
        return sloss.data * norm, sloss


# ============================================================================
# 训练 Worker
# ============================================================================

def train_worker(gpu, ngpus_per_node, vocab_src, vocab_tgt,
                 spacy_de, spacy_en, config, is_distributed=False):
    """在单个 GPU 上训练模型"""
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True

    if is_distributed:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group("nccl", init_method="env://", rank=gpu,
                                world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu, vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer, lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(), DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    """训练入口：单 GPU 或多 GPU 分布式训练"""
    if config.get("distributed", False):
        import torch.multiprocessing as mp
        ngpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        print(f"Number of GPUs detected: {ngpus}")
        mp.spawn(train_worker, nprocs=ngpus,
                 args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True))
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


def load_trained_model(vocab_src, vocab_tgt, model_path="multi30k_model_final.pt"):
    """加载已训练的模型"""
    if not exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# ============================================================================
# 简单复制任务（用于快速测试）
# ============================================================================

def data_gen(V, batch_size, nbatches):
    """生成用于 src-tgt 复制任务的随机数据"""
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


def run_simple_copy_task():
    """运行简单复制任务：用合成数据训练小 Transformer"""
    print("=" * 60)
    print("Running simple copy task (synthetic data)")
    print("=" * 60)

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer, lr_scheduler,
            mode="train",
        )
        model.eval()
        val_loss = run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(), DummyScheduler(),
            mode="eval",
        )[0]
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    result = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(f"\nInput:  {src.tolist()}")
    print(f"Output: {result.tolist()}")
    return model


# ============================================================================
# 评估：翻译结果检查
# ============================================================================

def check_outputs(valid_dataloader, model, vocab_src, vocab_tgt,
                  n_examples=15, pad_idx=2, eos_string="</s>"):
    """检查模型翻译结果"""
    results = []
    for idx in range(n_examples):
        print(f"\nExample {idx} " + "=" * 40)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx])
            .split(eos_string, 1)[0] + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results.append((rb, src_tokens, tgt_tokens, model_out, model_txt))
    return results


def run_evaluation(vocab_src, vocab_tgt, spacy_de, spacy_en,
                   model_path="multi30k_model_final.pt", n_examples=10):
    """运行评估：加载模型并展示翻译结果"""
    print("Preparing Data ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_dataloader = create_dataloaders(
        device, vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=1, is_distributed=False,
    )

    print("Loading Trained Model ...")
    model = load_trained_model(vocab_src, vocab_tgt, model_path)
    model = model.to(device)

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data