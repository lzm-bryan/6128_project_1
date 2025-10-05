import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python view_npz.py <your_file.npz>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading NPZ file: {path}")

# 载入 npz
data = np.load(path, allow_pickle=True)
print(f"\nKeys in file: {list(data.keys())}\n")

# 逐层查看
for key in data.files:
    arr = data[key]
    print(f"--- {key} ---")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")
    print("Preview:")
    print(arr[:5])  # 只显示前5行
    print()
