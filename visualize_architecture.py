"""
用 matplotlib 绘制 Transformer 完整数据流架构图
输出：transformer_dataflow.png

对应源码：transformer_model.py 中的 make_model / EncoderLayer / DecoderLayer
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


COLOR_ENC = "#4C78A8"
COLOR_ENC_LIGHT = "#D6E4F0"
COLOR_DEC = "#E45756"
COLOR_DEC_LIGHT = "#FADBD8"
COLOR_MEM = "#54A24B"
COLOR_MEM_LIGHT = "#D4EDDA"
COLOR_IO = "#8C8C8C"
COLOR_IO_LIGHT = "#EEEEEE"
COLOR_RES = "#B279A2"


def box(ax, x, y, w, h, text, face, edge, fontsize=9, weight="normal", zorder=2):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, facecolor=face, edgecolor=edge, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#222",
            zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.4, style="-", rad=0.0,
          shape="->", label=None, label_offset=(0.15, 0.0), label_fs=8,
          label_color="#555", zorder=3):
    conn = f"arc3,rad={rad}"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=shape, mutation_scale=14,
        linewidth=lw, color=color, linestyle=style,
        connectionstyle=conn, zorder=zorder,
    )
    ax.add_patch(a)
    if label:
        lx = (x1 + x2) / 2 + label_offset[0]
        ly = (y1 + y2) / 2 + label_offset[1]
        ax.text(lx, ly, label, fontsize=label_fs, color=label_color,
                ha="left", va="center", zorder=zorder + 1)


def draw_sublayer_conn(ax, cx, cy, w, h, sub_label, color_fill, color_edge,
                       section_title):
    """
    SublayerConnection 内部水平布局：
        (子框内左上角小标签：section_title)
        [LayerNorm] → [sublayer(...)] → [Dropout]  → ＋
    残差：左外侧竖线 → 上边横线 → 从上方进入 ＋
    cx, cy 为外框中心
    """
    outer = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.4, facecolor=color_fill, edgecolor=color_edge,
        zorder=2,
    )
    ax.add_patch(outer)

    ax.text(cx - w * 0.44, cy + h * 0.32, section_title,
            fontsize=7.8, color=color_edge, ha="left", va="center",
            fontweight="bold", zorder=6)

    y_row = cy - h * 0.12
    step_h = h * 0.42
    ln_w = w * 0.18
    sl_w = w * 0.36
    dp_w = w * 0.15
    ln_x = cx - w * 0.32
    sl_x = cx - w * 0.02
    dp_x = cx + w * 0.28

    box(ax, ln_x, y_row, ln_w, step_h, "LayerNorm",
        "white", color_edge, fontsize=7.8, zorder=3)
    box(ax, sl_x, y_row, sl_w, step_h, sub_label,
        "white", color_edge, fontsize=7.8, zorder=3)
    box(ax, dp_x, y_row, dp_w, step_h, "Dropout",
        "white", color_edge, fontsize=7.8, zorder=3)

    arrow(ax, ln_x + ln_w / 2, y_row, sl_x - sl_w / 2, y_row,
          color="#444", lw=1.0)
    arrow(ax, sl_x + sl_w / 2, y_row, dp_x - dp_w / 2, y_row,
          color="#444", lw=1.0)

    plus_x = cx + w * 0.44
    ax.text(plus_x, y_row, "＋", fontsize=14, ha="center", va="center",
            color=COLOR_RES, fontweight="bold", zorder=4)
    arrow(ax, dp_x + dp_w / 2, y_row, plus_x - 0.08, y_row,
          color="#444", lw=1.0)

    res_x = cx - w * 0.46
    res_bot = cy - h * 0.42
    res_top = cy + h * 0.14
    ax.plot([res_x, res_x], [res_bot, res_top],
            color=COLOR_RES, lw=1.1, linestyle=(0, (2, 2)), zorder=3)
    ax.plot([res_x, plus_x], [res_top, res_top],
            color=COLOR_RES, lw=1.1, linestyle=(0, (2, 2)), zorder=3)
    arrow(ax, plus_x, res_top, plus_x, y_row + 0.10,
          color=COLOR_RES, lw=1.1, style=(0, (2, 2)))
    ax.text(res_x - 0.04, cy - h * 0.16, "残差",
            fontsize=7, color=COLOR_RES, ha="right", va="center",
            rotation=90, zorder=4)


def draw_encoder_layer_expanded(ax, cx, cy, w, h):
    outer = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.8, facecolor=COLOR_ENC_LIGHT, edgecolor=COLOR_ENC,
        zorder=1,
    )
    ax.add_patch(outer)
    ax.text(cx, cy + h / 2 - 0.22, "EncoderLayer 1（完整展开）",
            fontsize=10.5, ha="center", va="center", fontweight="bold",
            color=COLOR_ENC, zorder=5,
            bbox=dict(boxstyle="round,pad=0.20",
                      facecolor="white", edgecolor=COLOR_ENC,
                      linewidth=0.8))

    sub_w = w * 0.88
    sub_h = h * 0.28
    y_bot = cy - h * 0.20
    y_top = cy + h * 0.16

    draw_sublayer_conn(ax, cx, y_bot, sub_w, sub_h,
                       "self_attn(x,x,x,src_mask)",
                       "#F7FBFF", COLOR_ENC,
                       "SublayerConnection[0]  ①self-attn")

    draw_sublayer_conn(ax, cx, y_top, sub_w, sub_h,
                       "FFN(512→2048→512)",
                       "#F7FBFF", COLOR_ENC,
                       "SublayerConnection[1]  ②FFN")

    arrow(ax, cx + sub_w * 0.44, y_bot + sub_h * 0.55,
          cx + sub_w * 0.44, y_top - sub_h * 0.55,
          color="#444", lw=1.2)


def draw_decoder_layer_expanded(ax, cx, cy, w, h, memory_x):
    outer = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.8, facecolor=COLOR_DEC_LIGHT, edgecolor=COLOR_DEC,
        zorder=1,
    )
    ax.add_patch(outer)
    ax.text(cx, cy + h / 2 - 0.24, "DecoderLayer 1（完整展开）",
            fontsize=10.5, ha="center", va="center", fontweight="bold",
            color=COLOR_DEC, zorder=5,
            bbox=dict(boxstyle="round,pad=0.20",
                      facecolor="white", edgecolor=COLOR_DEC,
                      linewidth=0.8))

    sub_w = w * 0.88
    sub_h = h * 0.22
    y1 = cy - h * 0.30
    y2 = cy - h * 0.02
    y3 = cy + h * 0.26

    draw_sublayer_conn(ax, cx, y1, sub_w, sub_h,
                       "self_attn(x,x,x,tgt_mask)",
                       "#FFF6F5", COLOR_DEC,
                       "SublayerConnection[0]  ①masked self-attn")

    draw_sublayer_conn(ax, cx, y2, sub_w, sub_h,
                       "src_attn(x, m, m, src_mask)",
                       "#FFF6F5", COLOR_DEC,
                       "SublayerConnection[1]  ②cross-attn (Q=x, K/V=m)")

    arrow(ax, memory_x, y2, cx - sub_w / 2 - 0.04, y2,
          color=COLOR_MEM, lw=1.8, rad=0.0,
          label="memory → K, V",
          label_offset=(-1.6, 0.20), label_fs=8.5, label_color=COLOR_MEM)

    draw_sublayer_conn(ax, cx, y3, sub_w, sub_h,
                       "FFN(512→2048→512)",
                       "#FFF6F5", COLOR_DEC,
                       "SublayerConnection[2]  ③FFN")

    arrow(ax, cx + sub_w * 0.44, y1 + sub_h * 0.55,
          cx + sub_w * 0.44, y2 - sub_h * 0.55,
          color="#444", lw=1.2)
    arrow(ax, cx + sub_w * 0.44, y2 + sub_h * 0.55,
          cx + sub_w * 0.44, y3 - sub_h * 0.55,
          color="#444", lw=1.2)


def draw_stacked_layer_box(ax, cx, cy, w, h, title, color_fill, color_edge,
                           subtitle="结构同上，参数独立"):
    for i, dx in enumerate([-0.08, -0.04, 0.0]):
        patch = FancyBboxPatch(
            (cx - w / 2 + dx, cy - h / 2 - dx * 0.5), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            linewidth=1.0, facecolor=color_fill, edgecolor=color_edge,
            alpha=0.55 + i * 0.15, zorder=1 + i,
        )
        ax.add_patch(patch)
    ax.text(cx, cy + 0.10, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color_edge, zorder=5)
    ax.text(cx, cy - 0.14, subtitle, ha="center", va="center",
            fontsize=8, color="#555", zorder=5)


def draw_simple_layer(ax, cx, cy, w, h, title, color_fill, color_edge):
    box(ax, cx, cy, w, h, title, color_fill, color_edge,
        fontsize=10, weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(20, 18), dpi=150)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(11, 29.2,
            "Transformer 完整数据流（Encoder / Decoder 每层内部展开）",
            fontsize=18, fontweight="bold", ha="center", color="#222")
    ax.text(11, 28.5,
            "make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8)",
            fontsize=10.5, ha="center", color="#666", family="monospace")

    ax.text(11, 27.0,
            "关键点：\n"
            "① EncoderLayer 内部 = 2 个 SublayerConnection（self-attn + FFN），"
            "对应 transformer_model.py:179 的 clones(..., 2)\n"
            "② DecoderLayer 内部 = 3 个 SublayerConnection（self-attn + cross-attn + FFN），"
            "对应 transformer_model.py:214 的 clones(..., 3)\n"
            "③ 每个 SublayerConnection = LayerNorm → sublayer(...) → Dropout → ＋残差 "
            "（transformer_model.py:50-51）\n"
            "④ memory 只在 Encoder 全部 6 层跑完后生成一份；Decoder 的每一层都通过 "
            "cross-attn 反复读取（K, V 来自 memory）\n"
            "⑤ 所有中间张量形状不变：Encoder 侧 [batch, seq_src, 512]，Decoder 侧 "
            "[batch, seq_tgt, 512]；只有 Generator 变成 [batch, seq_tgt, vocab_tgt]",
            fontsize=9.5, ha="center", va="center", color="#333",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7",
                      edgecolor="#CCC"))

    legend_x = 0.6
    legend_y = 23.5
    ax.text(legend_x, legend_y + 1.0, "图例：",
            fontsize=10, fontweight="bold", color="#333")
    legends = [
        (COLOR_ENC, "Encoder 通路（冷色）", "-"),
        (COLOR_DEC, "Decoder 通路（暖色）", "-"),
        (COLOR_MEM, "memory / cross-attn 通路（绿色）", "-"),
        (COLOR_RES, "残差连接（紫色虚线）", (0, (2, 2))),
        ("#333", "主数据流（实线粗箭头）", "-"),
    ]
    for i, (c, name, ls) in enumerate(legends):
        y = legend_y + 0.5 - i * 0.42
        ax.plot([legend_x, legend_x + 0.7], [y, y], color=c, lw=2.4,
                linestyle=ls)
        ax.text(legend_x + 0.85, y, name, fontsize=9, color="#333",
                va="center")

    enc_x = 5.0
    dec_x = 16.5
    memory_x = 9.5

    def draw_arrow_up(x, y_from, y_to, label=None, label_offset=(0.28, 0.0)):
        arrow(ax, x, y_from, x, y_to, color="#333", lw=1.7,
              label=label, label_offset=label_offset, label_fs=8.5)

    box(ax, enc_x, 1.0, 3.4, 0.7,
        '源句 "I love you"  →  id 序列', COLOR_IO_LIGHT, COLOR_IO, fontsize=9)
    ax.text(enc_x + 2.1, 1.0, "[64, 3]", fontsize=8, color="#777", va="center")

    box(ax, enc_x, 2.2, 3.4, 0.7, "Embeddings（查表）",
        COLOR_ENC_LIGHT, COLOR_ENC, fontsize=9.5)
    draw_arrow_up(enc_x, 1.4, 1.85, "[64, 3] → [64, 3, 512]")

    box(ax, enc_x, 3.4, 3.4, 0.7, "PositionalEncoding",
        COLOR_ENC_LIGHT, COLOR_ENC, fontsize=9.5)
    draw_arrow_up(enc_x, 2.6, 3.05)

    enc1_cy, enc1_h = 7.4, 5.6
    draw_encoder_layer_expanded(ax, enc_x, enc1_cy, 6.0, enc1_h)
    draw_arrow_up(enc_x, 3.8, enc1_cy - enc1_h / 2 - 0.05, "[64, 3, 512]")

    stack_enc_cy = 12.4
    draw_stacked_layer_box(ax, enc_x, stack_enc_cy, 4.4, 1.1,
                           "EncoderLayer 2 … 5",
                           COLOR_ENC_LIGHT, COLOR_ENC)
    draw_arrow_up(enc_x, enc1_cy + enc1_h / 2 + 0.05, stack_enc_cy - 0.7,
                  "[64, 3, 512]")

    enc6_cy = 14.3
    draw_simple_layer(ax, enc_x, enc6_cy, 4.4, 0.8,
                      "EncoderLayer 6", COLOR_ENC_LIGHT, COLOR_ENC)
    draw_arrow_up(enc_x, stack_enc_cy + 0.65, enc6_cy - 0.5)

    enc_norm_cy = 15.5
    box(ax, enc_x, enc_norm_cy, 3.6, 0.7, "LayerNorm（Encoder 顶部）",
        COLOR_ENC_LIGHT, COLOR_ENC, fontsize=9.5)
    draw_arrow_up(enc_x, enc6_cy + 0.5, enc_norm_cy - 0.45)

    memory_cy = 17.4
    box(ax, memory_x, memory_cy, 4.4, 1.1,
        "memory  =  Encoder 顶部输出\n[64, 3, 512]（一次生成，六次消费）",
        COLOR_MEM_LIGHT, COLOR_MEM, fontsize=10.5, weight="bold")
    arrow(ax, enc_x, enc_norm_cy + 0.4,
          memory_x - 2.2, memory_cy,
          color=COLOR_MEM, lw=1.8, rad=0.20)

    box(ax, dec_x, 1.0, 3.4, 0.7,
        '目标端 "我 爱"  →  id 序列', COLOR_IO_LIGHT, COLOR_IO, fontsize=9)
    ax.text(dec_x + 2.1, 1.0, "[64, 2]", fontsize=8, color="#777", va="center")

    box(ax, dec_x, 2.2, 3.4, 0.7, "Embeddings（查表）",
        COLOR_DEC_LIGHT, COLOR_DEC, fontsize=9.5)
    draw_arrow_up(dec_x, 1.4, 1.85, "[64, 2] → [64, 2, 512]")

    box(ax, dec_x, 3.4, 3.4, 0.7, "PositionalEncoding",
        COLOR_DEC_LIGHT, COLOR_DEC, fontsize=9.5)
    draw_arrow_up(dec_x, 2.6, 3.05)

    dec1_cy, dec1_h = 8.0, 7.0
    draw_decoder_layer_expanded(ax, dec_x, dec1_cy, 6.8, dec1_h,
                                memory_x=memory_x + 2.2)
    draw_arrow_up(dec_x, 3.8, dec1_cy - dec1_h / 2 - 0.05, "[64, 2, 512]")

    stack_dec_cy = 13.2
    draw_stacked_layer_box(ax, dec_x, stack_dec_cy, 4.8, 1.1,
                           "DecoderLayer 2 … 5",
                           COLOR_DEC_LIGHT, COLOR_DEC,
                           subtitle="结构同上；每层 cross-attn 都读同一份 memory")
    draw_arrow_up(dec_x, dec1_cy + dec1_h / 2 + 0.05, stack_dec_cy - 0.7,
                  "[64, 2, 512]")

    arrow(ax, memory_x + 2.2, memory_cy - 0.2,
          dec_x - 2.4, stack_dec_cy + 0.15,
          color=COLOR_MEM, lw=1.6, style=(0, (4, 3)), rad=-0.45,
          label="× 4 层  cross-attn\n都用同一份 memory",
          label_offset=(1.8, -1.2), label_fs=8.5, label_color=COLOR_MEM)

    dec6_cy = 15.1
    draw_simple_layer(ax, dec_x, dec6_cy, 4.8, 0.8,
                      "DecoderLayer 6", COLOR_DEC_LIGHT, COLOR_DEC)
    draw_arrow_up(dec_x, stack_dec_cy + 0.65, dec6_cy - 0.5)

    arrow(ax, memory_x + 2.2, memory_cy + 0.2,
          dec_x - 2.4, dec6_cy,
          color=COLOR_MEM, lw=1.4, style=(0, (4, 3)), rad=-0.45)

    dec_norm_cy = 16.3
    box(ax, dec_x, dec_norm_cy, 3.6, 0.7, "LayerNorm（Decoder 顶部）",
        COLOR_DEC_LIGHT, COLOR_DEC, fontsize=9.5)
    draw_arrow_up(dec_x, dec6_cy + 0.5, dec_norm_cy - 0.45)

    gen_cy = 17.7
    box(ax, dec_x, gen_cy, 4.4, 0.8,
        "Generator：Linear(512 → vocab_tgt) + log_softmax",
        "#FFEECC", "#D4A017", fontsize=9.5, weight="bold")
    draw_arrow_up(dec_x, dec_norm_cy + 0.45, gen_cy - 0.5, "[64, 2, 512]")

    out_cy = 19.0
    box(ax, dec_x, out_cy, 4.4, 0.7,
        "[64, 2, vocab_tgt]  →  argmax → 下一个词",
        "#FFEECC", "#D4A017", fontsize=9.5)
    draw_arrow_up(dec_x, gen_cy + 0.5, out_cy - 0.45)

    plt.tight_layout()
    out_path = "transformer_dataflow.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
