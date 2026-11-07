import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT = "docs/performance"
COLORS = {"muon": "#3B82F6", "adamw": "#F59E0B", "accent": "#10B981", "warn": "#EF4444"}


def style_ax(ax):
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
labels = ["Muon\nbatch 2", "AdamW\nbatch 2", "AdamW\nbatch 3"]
values = [1054, 1731, 2075]
colors = [COLORS["muon"], COLORS["adamw"], COLORS["adamw"]]
bars = ax.bar(labels, values, color=colors, width=0.55)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("tokens/s per Pod")
ax.set_title("Single-Pod throughput, dense05b, seq 1024", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(values) * 1.18)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/optimizer_throughput.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
labels = ["Muon", "AdamW"]
throughput = [1037, 1834]
loss = [4.3496, 3.6289]
x = np.arange(len(labels))
width = 0.34
b1 = ax.bar(x - width / 2, throughput, width, label="tokens/s", color=COLORS["accent"])
ax2 = ax.twinx()
b2 = ax2.bar(x + width / 2, loss, width, label="final loss", color=COLORS["warn"])
for bar, value in zip(b1, throughput):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 30, f"{value:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, value in zip(b2, loss):
    ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("tokens/s")
ax2.set_ylabel("final loss")
ax.set_xticks(x, labels)
ax.set_title("Same ~205K tokens: throughput vs loss", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(throughput) * 1.22)
ax2.set_ylim(0, max(loss) * 1.25)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax2.legend(loc="upper right", frameon=False, fontsize=9)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/throughput_vs_loss.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
labels = ["Step window\n(4 Pod)", "End-to-end\n(4 Pod, 50 steps)"]
mfu = [44.5, 22.0]
bars = ax.bar(labels, mfu, color=[COLORS["accent"], COLORS["warn"]], width=0.5)
for bar, value in zip(bars, mfu):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(100, color="#6B7280", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(1.42, 100, "100% peak", va="center", fontsize=8, color="#6B7280")
ax.set_ylabel("MFU (%)")
ax.set_title("Measured MFU under two accounting windows", fontsize=11, fontweight="bold")
ax.set_ylim(0, 108)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/mfu_windows.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
ns = [1, 2, 3, 5]
ms = [263.08, 427.98, 593.99, 926.21]
ax.plot(ns, ms, marker="o", color=COLORS["muon"], linewidth=2, markersize=7)
for x, y in zip(ns, ms):
    ax.text(x, y + 22, f"{y:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Newton–Schulz iterations")
ax.set_ylabel("optimizer median (ms)")
ax.set_title("Muon NS iterations vs optimizer cost", fontsize=11, fontweight="bold")
ax.set_xticks(ns)
ax.set_ylim(0, max(ms) * 1.18)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/muon_ns_cost.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
parts = ["SwiGLU\n(up + down)", "Embed +\nlm_head", "QKV + Wo", "Attention\nQK/AV"]
flops = [51, 26, 23, 8]
colors = [COLORS["muon"], COLORS["adamw"], COLORS["accent"], COLORS["warn"]]
bars = ax.barh(parts[::-1], flops[::-1], color=colors[::-1], height=0.55)
for bar, value in zip(bars, flops[::-1]):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value}%",
            va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Share of training FLOP per token (%)")
ax.set_title("dense05b FLOP breakdown, seq 1024", fontsize=11, fontweight="bold")
ax.set_xlim(0, 60)
ax.grid(axis="x", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/flop_breakdown.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.6))
labels = ["4-Pod\ntheoretical peak", "4-Pod\nmeasured step", "4-Pod\nend-to-end", "24h target"]
toks = [24500, 10900, 5300, 1297000]
bars = ax.bar(labels, toks, color=[COLORS["accent"], COLORS["muon"], COLORS["adamw"], COLORS["warn"]], width=0.55)
for bar, value in zip(bars, toks):
    ax.text(bar.get_x() + bar.get_width() / 2, value * 1.04, f"{value:,.0f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_yscale("log")
ax.set_ylabel("aggregate tokens/s (log scale)")
ax.set_title("Throughput vs 24-hour corpus target", fontsize=11, fontweight="bold")
ax.tick_params(axis="x", labelsize=8)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/throughput_vs_target.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.8))
ns_labels = ["NS = 2", "NS = 3"]
step_s = [1.085, 1.225]
eval_loss = [4.1058, 4.5672]
sizes = [180, 180]
colors = [COLORS["accent"], COLORS["warn"]]
ax.scatter(step_s, eval_loss, s=sizes, c=colors, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
for x, y, label in zip(step_s, eval_loss, ns_labels):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(12, 10),
                fontsize=10, fontweight="bold")
ax.set_xlabel("steady-state step time (s), lower is better")
ax.set_ylabel("eval loss, lower is better")
ax.set_title("Muon NS trade-off: NS=2 is faster and lower-loss", fontsize=11, fontweight="bold")
ax.set_xlim(0.98, 1.32)
ax.set_ylim(3.95, 4.72)
ax.text(0.99, 4.62, "better", fontsize=9, color="#6B7280", style="italic")
ax.annotate("", xy=(1.02, 4.06), xytext=(1.28, 4.60),
            arrowprops=dict(arrowstyle="->", color="#9CA3AF", linewidth=1.2, linestyle="--"))
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/ns_quality_speed.png", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6.4, 3.8))
ns_vals = [1, 2, 3, 5]
toks = [2440, 1730, 1522, 1214]
evals = [7.4157, 7.3356, 7.1983, 7.0768]
ax.plot(ns_vals, toks, marker="o", color=COLORS["accent"], linewidth=2,
        markersize=7, label="tok/s (left)")
ax.set_xlabel("Newton–Schulz iterations")
ax.set_ylabel("step-window tok/s", color=COLORS["accent"])
ax.tick_params(axis="y", labelcolor=COLORS["accent"])
ax2 = ax.twinx()
ax2.plot(ns_vals, evals, marker="s", color=COLORS["warn"], linewidth=2,
         markersize=7, label="eval loss (right)")
ax2.set_ylabel("eval loss @ 500 steps", color=COLORS["warn"])
ax2.tick_params(axis="y", labelcolor=COLORS["warn"])
for x, y in zip(ns_vals, toks):
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, fontweight="bold", color=COLORS["accent"])
for x, y in zip(ns_vals, evals):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14),
                 ha="center", fontsize=8, fontweight="bold", color=COLORS["warn"])
ax.set_title("NS iterations: throughput vs eval loss (500 steps)", fontsize=11, fontweight="bold")
ax.set_xticks(ns_vals)
ax.grid(alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/ns_tradeoff_curve.png", bbox_inches="tight")
plt.close(fig)

print("charts written to", OUT)
