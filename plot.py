import pandas as pd
import matplotlib.pyplot as plt

# ============ 1. 基本配置 ============
CSV_PATH = "main/020/020光纤.csv"        # CSV 文件路径
COLUMN_NAME = "Fiber_s5"        # 要绘制的列名
ENCODING = "utf-8"           # 若中文乱码可改成 "gbk" 或 "utf-8-sig"

# ============ 2. 读取数据 ============
df = pd.read_csv(CSV_PATH, encoding=ENCODING)

if COLUMN_NAME not in df.columns:
    raise ValueError(f"列名 '{COLUMN_NAME}' 不存在，现有列：{list(df.columns)}")

# 取出该列数据（自动忽略空值）
series = pd.to_numeric(df[COLUMN_NAME], errors="coerce").dropna().reset_index(drop=True)

# ============ 3. 绘图 ============
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False   # 正常显示负号

plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)

plt.title(f"列「{COLUMN_NAME}」数据折线图")
plt.xlabel("序号（从头到尾）")
plt.ylabel(COLUMN_NAME)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()