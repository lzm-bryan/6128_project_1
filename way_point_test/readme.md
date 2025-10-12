Overview

This folder contains scripts for visualizing indoor-location data using both Folium (HTML maps) and Matplotlib (static figures).

Scripts

test.py
Visualize all waypoints in site1/B1.

plot.py
Render the base geomap.

indoor.py
Naive overlay without coordinate mapping. The Folium output looks odd due to missing transforms.

indoor_plus.py
Apply coordinate-to-map JSON mapping, then render with Folium. The map looks correct after simple flips because the dataset’s coordinate system is already aligned.

indoor_matplot_simple.py
Minimal Matplotlib plotting. Current figures are ugly; tune parameters (figure size, linewidth, marker size, alpha, axis limits, aspect, fonts) to improve aesthetics.

geomag_heatnmap.py
Heatmap attempt v1. Abandoned.

geomag_trackmap.py
Generate paper-style track figures.

geomag_trackmap_folium.py
Folium version of the track map. Output HTML is very large.

geomag_trackmap_plus.py
Slightly optimized track figures. Further optimization TBD.

Batch runs
Several scripts support .bat batch execution.

picture.py
Take screenshots of HTML maps. Capture every HTML produced by indoor_plus.py (currently under folium_maps/) and save images to map_screenshots/.

这里边进行画图

test.py把B1中所有waypoint展示一下
plot.py把geomap展示一下
indoor.py单纯结合没有映射 生成的图(folium)很奇怪
indoor_plus.py把坐标和map的json进行映射了,生成的图(folium)很好看了 (其实映射就是翻转一些，因为数据集坐标系已经是对齐了的)
(bat文件批量)
indoor_matplot_simple(matplot作图代码)但是现在的图好丑，可能要调一下参数

geomag_heatnmap热力图尝试1(放弃了)
geomag_trackmap参照论文格式生成图片
geomag_trackmap_folium生成folium（但是文件太大了）
geomag_trackmap_plus优化一点图片(要怎么弄？还没想好怎么优化)
(bat文件批量)

picture.py html截一下图 把indoor_plus生成的html(现在放在folium_maps中)都截一个图保存到map_screenshots文件夹