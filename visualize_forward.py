"""
MultiHeadedAttention.forward() Visualization
============================================
Step-by-step matplotlib diagrams for the forward pass
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_box(ax, xy, wh, text, color='#E3F2FD', fontsize=10, bold=False):
    box = FancyBboxPatch(xy, wh[0], wh[1],
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0] + wh[0]/2, xy[1] + wh[1]/2, text,
            ha='center', va='center', fontsize=fontsize, fontweight=weight)

def draw_arrow(ax, start, end, color='#555555', label=''):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                connectionstyle='arc3,rad=0'))
    if label:
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(mid[0]+0.15, mid[1]+0.1, label, fontsize=8, color=color, style='italic')

# ============================================================
# Figure 1: Overall forward flow
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('MultiHeadedAttention.forward() -- Complete Flow', fontsize=16, fontweight='bold', pad=20)

# Inputs
draw_box(ax, (0.5, 10.5), (2.5, 0.7), 'query\n[batch, seq, 512]', '#FFE0B2')
draw_box(ax, (3.5, 10.5), (2.5, 0.7), 'key\n[batch, seq, 512]', '#FFE0B2')
draw_box(ax, (6.5, 10.5), (2.5, 0.7), 'value\n[batch, seq, 512]', '#FFE0B2')
draw_box(ax, (9.5, 10.5), (2.5, 0.7), 'mask\n[batch, seq, seq]', '#E0E0E0')

ax.text(1.75, 10.0, 'e.g. "I love Beijing"', ha='center', fontsize=8, color='#888888')

# Step 1: Linear projection
draw_box(ax, (0.5, 8.5), (3.5, 0.8),
         'Step 1: Linear Project\n  W_Q: 512->512  W_K: 512->512\n  W_V: 512->512', '#FFF9C4')
draw_arrow(ax, (2.0, 10.5), (2.0, 9.3))
draw_arrow(ax, (5.0, 10.5), (3.0, 9.0))
draw_arrow(ax, (8.0, 10.5), (3.0, 8.9))

# mask unsqueeze
draw_box(ax, (9.5, 8.5), (2.8, 0.8), 'mask.unsqueeze(1)\n-> [batch, 1, seq, seq]', '#E0E0E0')
draw_arrow(ax, (10.75, 10.5), (10.9, 9.3))

# Step 2: Split into heads
draw_box(ax, (0.5, 6.5), (3.5, 1.0),
         'Step 2: Split into h=8 heads\n  .view(batch, -1, 8, 64)\n  .transpose(1, 2)\n-> [batch, 8, seq, 64]', '#C8E6C9')
draw_arrow(ax, (2.0, 8.5), (2.0, 7.5))

draw_box(ax, (5.0, 6.5), (6.5, 1.0),
         'Q: [batch, 8, seq_q, 64]\nK: [batch, 8, seq_k, 64]\nV: [batch, 8, seq_v, 64]\n(8 heads, each gets 64-dim)', '#C8E6C9')

# Step 3: Per-head attention
draw_box(ax, (0.5, 4.5), (11.5, 1.3),
         'Step 3: 8 heads compute Scaled Dot-Product Attention in parallel\n'
         '  head0: scores = Q0*K0T/sqrt(64) -> softmax -> *V0    head1: same for Q1,K1,V1\n'
         '  head2: ...    head3: ...    ...    head7: ...\n'
         '-> each head output [batch, seq, 64]   +   attn weights [batch, 8, seq, seq] stored in self.attn',
         '#BBDEFB')
draw_arrow(ax, (2.0, 6.5), (2.0, 5.8))

# Step 4: Concat
draw_box(ax, (0.5, 2.5), (5.0, 1.0),
         'Step 4: Concatenate 8 heads\n  .transpose(1,2).view(batch, -1, 8*64)\n-> [batch, seq, 512]', '#FFCCBC')
draw_arrow(ax, (2.0, 4.5), (2.0, 3.5))

# Step 5: Output projection
draw_box(ax, (0.5, 0.8), (5.0, 0.8),
         'Step 5: W_O projection: 512 -> 512\n-> final output [batch, seq, 512]', '#CE93D8')
draw_arrow(ax, (2.0, 2.5), (2.0, 1.6))

# Side note
draw_box(ax, (12.5, 0.8), (3.0, 5.0),
         'Key Idea:\n\n8 "experts" each\nview the same sentence\nfrom a different angle:\n\nhead0: subject-verb\nhead1: modifiers\nhead2: coreference\n...\nhead7: word order\n\nW_O merges all\ntheir findings.',
         '#F3E5F5')

plt.tight_layout()
plt.savefig('/home/root123/workspace/source/annotated-transformer-opt/forward_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: forward_overview.png")


# ============================================================
# Figure 2: Scaled Dot-Product Attention step-by-step
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Scaled Dot-Product Attention -- Step by Step', fontsize=16, fontweight='bold', y=0.98)

np.random.seed(42)
seq_len = 3
dk = 4
Q = np.random.randn(seq_len, dk)
K = np.random.randn(seq_len, dk)
V = np.random.randn(seq_len, dk)
tokens = ['I', 'love', 'Beijing']

# Subplot 1: Q
ax = axes[0, 0]
im = ax.imshow(Q, cmap='YlOrRd', aspect='auto')
ax.set_title('Step 1: Query matrix Q', fontsize=12, fontweight='bold')
ax.set_xticks(range(dk)); ax.set_xticklabels([f'd{i}' for i in range(dk)], fontsize=7)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
for i in range(seq_len):
    for j in range(dk):
        ax.text(j, i, f'{Q[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)

# Subplot 2: K
ax = axes[0, 1]
im = ax.imshow(K, cmap='YlOrRd', aspect='auto')
ax.set_title('Key matrix K', fontsize=12, fontweight='bold')
ax.set_xticks(range(dk)); ax.set_xticklabels([f'd{i}' for i in range(dk)], fontsize=7)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
for i in range(seq_len):
    for j in range(dk):
        ax.text(j, i, f'{K[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)

# Subplot 3: Q * K^T
ax = axes[0, 2]
scores = Q @ K.T
vmax = max(abs(scores.min()), abs(scores.max()))
im = ax.imshow(scores, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_title('Step 2: Q * K^T (raw scores)', fontsize=12, fontweight='bold')
ax.set_xticks(range(seq_len)); ax.set_xticklabels(tokens, fontsize=10)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
ax.set_xlabel('Key (from)', fontsize=10); ax.set_ylabel('Query (to)', fontsize=10)
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# Subplot 4: scale
ax = axes[1, 0]
scaled = scores / np.sqrt(dk)
vmax2 = max(abs(scaled.min()), abs(scaled.max()))
im = ax.imshow(scaled, cmap='coolwarm', aspect='auto', vmin=-vmax2, vmax=vmax2)
ax.set_title('Step 3: Scale by sqrt(dk)=2', fontsize=12, fontweight='bold')
ax.set_xticks(range(seq_len)); ax.set_xticklabels(tokens, fontsize=10)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
ax.set_xlabel('Key', fontsize=10); ax.set_ylabel('Query', fontsize=10)
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{scaled[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# Subplot 5: softmax
ax = axes[1, 1]
attn_weights = np.exp(scaled) / np.exp(scaled).sum(axis=-1, keepdims=True)
im = ax.imshow(attn_weights, cmap='YlOrRd', aspect='auto')
ax.set_title('Step 4: Softmax (rows sum to 1)', fontsize=12, fontweight='bold')
ax.set_xticks(range(seq_len)); ax.set_xticklabels(tokens, fontsize=10)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
ax.set_xlabel('Key', fontsize=10); ax.set_ylabel('Query', fontsize=10)
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{attn_weights[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# Subplot 6: * V
ax = axes[1, 2]
output = attn_weights @ V
im = ax.imshow(output, cmap='YlGnBu', aspect='auto')
ax.set_title('Step 5: Attn Weights * V', fontsize=12, fontweight='bold')
ax.set_xticks(range(dk)); ax.set_xticklabels([f'd{i}' for i in range(dk)], fontsize=7)
ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=10)
for i in range(seq_len):
    for j in range(dk):
        ax.text(j, i, f'{output[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('/home/root123/workspace/source/annotated-transformer-opt/attention_steps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: attention_steps.png")


# ============================================================
# Figure 3: Shape transformation (query example)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Shape Transformation of query (d_model=512, h=8, d_k=64)', fontsize=15, fontweight='bold', pad=20)

# Input
draw_box(ax, (0.5, 5.0), (3.0, 1.0),
         'query\nshape: [batch, seq, 512]\nmeaning: each token = 512-dim vector', '#FFE0B2', fontsize=10)
ax.text(2.0, 4.7, 'example: [2, 10, 512]', ha='center', fontsize=8, color='#888')
ax.text(2.0, 4.4, '2 sentences, 10 tokens, 512-dim each', ha='center', fontsize=8, color='#888')

# Linear projection
draw_box(ax, (5.0, 5.0), (3.0, 1.0),
         'self.linears[0](query)\nshape: [batch, seq, 512]\nmeaning: W_Q linear transform', '#FFF9C4', fontsize=10)
draw_arrow(ax, (3.5, 5.5), (5.0, 5.5), label='linear[0]')

# view
draw_box(ax, (9.5, 5.0), (3.0, 1.0),
         '.view(batch, -1, 8, 64)\nshape: [batch, seq, 8, 64]\nmeaning: split into 8 heads', '#C8E6C9', fontsize=10)
draw_arrow(ax, (8.0, 5.5), (9.5, 5.5), label='view')

# transpose
draw_box(ax, (9.5, 2.5), (3.0, 1.0),
         '.transpose(1, 2)\nshape: [batch, 8, seq, 64]\nmeaning: head dimension first', '#BBDEFB', fontsize=10)
draw_arrow(ax, (11.0, 5.0), (11.0, 3.5), label='transpose')

# Visual blocks
ax.text(3.0, 2.5, 'Visualization:', fontsize=10, fontweight='bold', color='#333')

# Original [batch, seq, 512]
ax.text(0.5, 2.0, '[batch, seq, 512]', fontsize=9, fontweight='bold')
rect = plt.Rectangle((0.5, 0.5), 3.5, 1.0, facecolor='#FFE0B2', edgecolor='#333', lw=1.5)
ax.add_patch(rect)
ax.text(2.25, 1.0, 'seq * 512', ha='center', va='center', fontsize=9)

# Split: [batch, seq, 8, 64]
ax.text(5.0, 2.0, '[batch, seq, 8, 64]', fontsize=9, fontweight='bold')
for i in range(8):
    c = plt.cm.Set3(i/8)
    rect = plt.Rectangle((5.0 + i*0.5, 0.5), 0.45, 1.0, facecolor=c, edgecolor='#333', lw=0.8)
    ax.add_patch(rect)
    ax.text(5.0 + i*0.5 + 0.22, 0.2, f'h{i}', ha='center', fontsize=6)

# After transpose: [batch, 8, seq, 64]
ax.text(10.0, 2.0, '[batch, 8, seq, 64]', fontsize=9, fontweight='bold')
for i in range(8):
    c = plt.cm.Set3(i/8)
    rect = plt.Rectangle((10.0 + i*0.5, 0.5), 0.45, 1.0, facecolor=c, edgecolor='#333', lw=0.8)
    ax.add_patch(rect)
    ax.text(10.0 + i*0.5 + 0.22, 0.2, f'h{i}', ha='center', fontsize=6)

# Arrows
ax.annotate('', xy=(5.0, 1.0), xytext=(4.0, 1.0),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
ax.annotate('', xy=(10.0, 1.0), xytext=(9.0, 1.0),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

plt.tight_layout()
plt.savefig('/home/root123/workspace/source/annotated-transformer-opt/shape_transform.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: shape_transform.png")


# ============================================================
# Figure 4: Concat + Output projection
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Concat Heads + Output Projection (Steps 4 + 5)', fontsize=15, fontweight='bold', pad=20)

# 8 heads
ax.text(0.5, 6.0, '8 head outputs:', fontsize=11, fontweight='bold')
for i in range(8):
    c = plt.cm.Set3(i/8)
    rect = plt.Rectangle((0.5 + i*1.0, 4.5), 0.9, 1.0, facecolor=c, edgecolor='#333', lw=1.2)
    ax.add_patch(rect)
    ax.text(0.5 + i*1.0 + 0.45, 5.0, f'head{i}\n[batch, seq, 64]', ha='center', fontsize=8)

# Concat
draw_box(ax, (0.5, 2.8), (8.5, 0.8),
         '.transpose(1,2).contiguous().view(batch, -1, 512)\n-> [batch, seq, 512]   concat back to 512-dim',
         '#FFCCBC', fontsize=10)
draw_arrow(ax, (4.5, 4.5), (4.5, 3.6), label='concat')

# Output projection
draw_box(ax, (0.5, 1.0), (6.0, 0.9),
         'self.linears[-1](x) = W_O projection: 512 -> 512\n-> final output [batch, seq, 512]',
         '#CE93D8', fontsize=10)
draw_arrow(ax, (4.5, 2.8), (4.5, 1.9), label='linear[3]')

# Side note
draw_box(ax, (10.5, 1.0), (4.5, 5.0),
         'Role of W_O:\n\n8 heads compute independently,\nbut W_O fuses information\nfrom different perspectives\ninto one unified representation.\n\nLike 8 experts each give\ntheir report, then W_O\nsynthesizes everything.',
         '#F3E5F5', fontsize=10)

plt.tight_layout()
plt.savefig('/home/root123/workspace/source/annotated-transformer-opt/concat_output.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: concat_output.png")

print("\nAll 4 diagrams generated successfully!")