"""Generate publication figures from recorded results, with no external assets."""

import csv
import json
from pathlib import Path
import os

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from ozon_dimensioner.simulation import SENSORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
BLUE = "#005bff"
DARK = "#182b49"
CYAN = "#20a5a5"
ORANGE = "#df7c29"

fig, ax = plt.subplots(figsize=(10, 5.4))
ax.add_patch(
    Rectangle(
        (-1250, -55), 2500, 1660, fill=False, edgecolor="#8797af", linestyle="--", lw=1
    )
)
ax.text(800, 1550, "Защитный кожух", fontsize=8, color="#62718a")
ax.add_patch(Rectangle((-300, -35), 600, 35, color=DARK))
ax.add_patch(
    Rectangle(
        (-300, 0), 600, np.sqrt(340000), facecolor=BLUE, alpha=0.09, edgecolor=BLUE
    )
)
ax.add_patch(Rectangle((-100, 0), 200, 250, facecolor=CYAN, alpha=0.4))
for s in SENSORS:
    ax.scatter(s.x, s.z, s=120, color=BLUE, zorder=5)
    ax.plot([s.x, s.target_x], [s.z, s.target_z], "--", color=BLUE, lw=1)
    ax.annotate(
        f"{s.name}: ({s.x:.0f}, {s.z:.0f}) мм",
        (s.x, s.z),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.annotate("Расчётная область 600 × 583 мм", (-290, 540), fontsize=9)
ax.annotate("Лента 600 мм", (0, -90), ha="center")
ax.set(
    xlim=(-1350, 1350),
    ylim=(-150, 1650),
    xlabel="Поперёк ленты x, мм",
    ylabel="Высота z, мм",
)
ax.set_title(
    "Поперечное сечение измерительного поста", loc="left", fontweight="bold", color=DARK
)
ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig(OUT / "station.png", dpi=190)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set(xlim=(-450, 560), ylim=(-210, 210))
ax.add_patch(Rectangle((-300, -190), 600, 380, facecolor="#eef3fa", edgecolor=DARK))
for s, c in zip(SENSORS, [BLUE, CYAN, ORANGE]):
    ax.plot([-300, 300], [s.plane_y, s.plane_y], color=c, lw=2)
    ax.text(310, s.plane_y, f"{s.name}: y={s.plane_y:.0f}", va="center", fontsize=9)
ax.annotate(
    "Движение 1 м/с",
    xy=(-365, 140),
    xytext=(-365, -140),
    arrowprops={"arrowstyle": "->", "color": DARK},
    ha="center",
    rotation=90,
)
ax.plot([-300, 300], [-160, -160], ":", color="#62718a")
ax.scatter([-300, 300], [-160, -160], marker="s", s=22, color="#62718a")
ax.text(310, -160, "Триггер, z=5 мм", fontsize=8, va="center")
ax.set_xlabel("x, мм")
ax.set_ylabel("y, мм")
ax.set_title(
    "Вид сверху: плоскости сканирования разнесены на 100 мм",
    loc="left",
    fontweight="bold",
    color=DARK,
)
fig.tight_layout()
fig.savefig(OUT / "station_top.png", dpi=190)
plt.close(fig)

payloads = json.loads((ROOT / "results/measurements.json").read_text())
fig = plt.figure(figsize=(10, 4.4))
for col, index in enumerate([13, 33], 1):
    data = np.load(ROOT / f"results/cloud_{index:03d}.npz")
    p = data["points"]
    ids = data["sensor_ids"]
    stride = max(1, len(p) // 5000)
    ax = fig.add_subplot(1, 2, col, projection="3d")
    ax.scatter(*p[::stride].T, c=ids[::stride], s=1, cmap="viridis", alpha=0.45)
    b = payloads[index]["candidate_box"]
    ext = np.array(b["dimensions_mm"])
    axes = np.array(b["axes"])
    center = np.array(b["center_mm"])
    from itertools import product

    c = np.array(list(product([-0.5, 0.5], repeat=3))) * ext @ axes.T + center
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                ax.plot(*np.vstack([c[i], c[j]]).T, color=BLUE, lw=1.3)
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    ax.set_title(
        f"Сценарий {index}: " + ("коробка" if index == 13 else "Г-образный товар"),
        fontsize=10,
    )
    ax.set_box_aspect(np.maximum(np.ptp(p, axis=0), 40))
fig.tight_layout()
fig.savefig(OUT / "demo.png", dpi=190)
plt.close(fig)

rows = list(csv.DictReader((ROOT / "results/benchmark.csv").open()))
normal = [r for r in rows if r["group"] in ("nominal", "edge", "curved", "dropout")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
x = np.arange(len(normal))
errors = [float(r["max_normalized_error"]) for r in normal]
axes[0].bar(x, errors, color=BLUE, width=0.85)
axes[0].axhline(1, color=ORANGE, ls="--", label="Граница допуска")
axes[0].set(
    xlabel="Сценарий с эталоном", ylabel="Максимальная ошибка / допуск", ylim=(0, 1.1)
)
axes[0].legend(fontsize=8)
axes[1].plot([float(r["processing_ms"]) for r in rows], color=BLUE, marker=".", lw=1)
axes[1].set(xlabel="Сценарий", ylabel="Обработка облака, мс")
axes[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "metrics.png", dpi=190)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")
ax.set(xlim=(0, 10), ylim=(0, 3))
labels = [
    "Профили +\nэнкодер",
    "Общие\nкоординаты",
    "Лента +\nконтроль данных",
    "Оболочка +\nпоиск OBB",
    "Контроль\nдостоверности",
    "Outbox\n→ WMS",
]
for i, label in enumerate(labels):
    x = 0.1 + i * 1.66
    ax.add_patch(
        Rectangle((x, 1.1), 1.45, 0.8, facecolor="#edf3ff", edgecolor=BLUE, lw=1.2)
    )
    ax.text(x + 0.725, 1.5, label, ha="center", va="center", fontsize=9)
    if i < 5:
        ax.add_patch(
            FancyArrowPatch(
                (x + 1.45, 1.5),
                (x + 1.66, 1.5),
                arrowstyle="->",
                mutation_scale=12,
                color=DARK,
            )
        )
ax.text(
    0.1,
    0.55,
    "Каждый результат: ID, миллиметры, статус, версия алгоритма и калибровки",
    fontsize=10,
    color=DARK,
)
fig.tight_layout()
fig.savefig(OUT / "pipeline.png", dpi=190)
plt.close(fig)
print("Figures written:", len(list(OUT.glob("*.png"))))
