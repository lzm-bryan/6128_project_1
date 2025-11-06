
Folders

NN/: fingerprint-related code.

way_point_test/: plotting for true coordinates and magnetic-field heatmaps.

indoor_cache/: dataset downloaded from the website.

fetch_test/: data-download test scripts.

Data & Mapping

Waypoints are in meters.

The JSON files already define the image domain.

You can plot directly: use waypoints as coordinates, but apply a flip so the orientation matches the floor image.

Scripts

main_heat.py: builds a full Folium page with waypoints and heatmaps.
Initially had no mapping so overlays were misaligned; a flip was added and alignment is now correct.

Output

multi_floor_gt_heat.html: final web visualization.

# 6128
fetch_test是下载数据代码
way_point_test是真实坐标和磁场热力图绘图相关代码
NN文件夹是fingerprint相关代码

对数据:waypoint是米级别坐标; json文件已经给定了图片的定义域
所以直接作图，坐标是waypoint，但是需要flip
main_heat.py是画一个完整的folium网站包括坐标和热力图(增加了flip)
multi_floor_gt_heat.html是网页端可视化文件
