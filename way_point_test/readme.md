这里边进行画图

test.py把B1中所有waypoint展示一下
plot.py把geomap展示一下
indoor.py单纯结合没有映射 生成的图(folium)很奇怪
indoor_plus.py把坐标和map的json进行映射了,生成的图(folium)很好看了
(bat文件批量)

geomag_heatnmap热力图尝试1(放弃了)
geomag_trackmap参照论文格式生成图片
geomag_trackmap_folium生成folium（但是文件太大了）
geomag_trackmap_plus优化一点图片(要怎么弄？还没想好怎么优化)
(bat文件批量)

picture.py html截一下图 把indoor_plus生成的html(现在放在folium_maps中)都截一个图保存到map_screenshots文件夹