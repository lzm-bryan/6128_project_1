Data & Outputs

site1, site2: downloaded raw data.

transform: analysis of model printouts on site1/B1 (model outputs inspected here).

*.csv: produced by the command that consolidates the TXT files under each site*/floor into a single CSV.

out* folders: packaged datasets (train/val/test etc.) created by the command below.

Workflow

Quick structure check in Excel
Use txt2excel_sensors to extract feature values X and generate an .xlsx preview.
Note: .xlsx is just for looking; it’s less convenient than .csv. Use the CSV generated in the next step for real work.

Build labels + features CSV
Run prep_fingerprint_csv to create a labeled feature CSV.
You can control data granularity. --dense means more samples.
Supports batch runs via .bat.

Package NumPy datasets
Run preprocess_fingerprint_dataset to generate .npy bundles for train/val/test.
If you used dense above, use dense here as well.
Supports batch runs via .bat.

Train models

train_mlp: trains a baseline MLP, saves predictions and labels for comparison.

train_stronger: stronger model using a Transformer. Produces test_predictions.csv.

Visualize results

viz_pred_vs_true: plot predicted vs. true values to inspect performance.

Requirements
conda install scikit-learn
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

这个文件夹进行深度学习

site1 site2文件夹是下载下来的数据
transform文件夹是模型在site1 b1上的输出打印值的数据分析
csv文件是下边命令整理的site中的txt文件
out开头文件夹是下边命令把csv文件封装训练集验证集测试集等数据

先提取出来数据excel看看格式
txt2excel_sensors看着搞一下特征值x 生成xlsx（xlsx不如csv好用，用下边csv）

prep_fingerprint_csv标签和特征值csv 生成csv（可以控制数据的粒度 dense是更多数据）
（bat文件批运行）
preprocess_fingerprint_dataset生成npy的训练数据测试数据包（如果上边是dense这里也是dense）
（bat文件批运行）

train_mlp训练保存预测和标签看对比 一个MLP模型
train_stronger更强模型 使用transformer
会生成test_predictions.csv文件
viz_pred_vs_true看一下预测值的结果怎么样 画图


requirement
conda install scikit-learn
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
