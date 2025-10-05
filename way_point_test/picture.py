# 默认大小
# from playwright.sync_api import sync_playwright
# import os, time
#
# html_dir = "folium_maps"
# output_dir = "map_screenshots"
# os.makedirs(output_dir, exist_ok=True)
#
# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=True)
#     page = browser.new_page()
#     page.set_viewport_size({"width": 1600, "height": 1200})
#
#     for file in os.listdir(html_dir):
#         if file.endswith(".html"):
#             html_path = os.path.abspath(os.path.join(html_dir, file))
#             out_path = os.path.join(output_dir, file.replace(".html", ".png"))
#             page.goto(f"file:///{html_path}")
#             time.sleep(2)
#             page.screenshot(path=out_path)
#             print(f"✅ Saved: {out_path}")
#
#     browser.close()

# 放大一下
from playwright.sync_api import sync_playwright
import os, time

html_dir = "folium_maps"
output_dir = "map_screenshots"
os.makedirs(output_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--force-device-scale-factor=2",  # 高清
            "--no-sandbox"
        ]
    )

    page = browser.new_page()
    page.set_viewport_size({"width": 1200, "height": 800})

    for file in os.listdir(html_dir):
        if not file.endswith(".html"):
            continue

        html_path = os.path.abspath(os.path.join(html_dir, file))
        out_path = os.path.join(output_dir, file.replace(".html", ".png"))
        print(f"[+] Opening {file}")

        # 打开 HTML
        page.goto(f"file:///{html_path}")
        page.wait_for_timeout(1000)  # 等地图加载

        # ✅ 1. 模拟点击“放大”按钮（Leaflet 默认类名 leaflet-control-zoom-in）
        try:
            zoom_in = page.query_selector(".leaflet-control-zoom-in")
            if zoom_in:
                zoom_in.click()
                zoom_in.click()
                print("🔍 Clicked zoom in")
                page.wait_for_timeout(500)
        except Exception as e:
            print(f"⚠️ Zoom click failed: {e}")

        # ✅ 2. 再截图
        try:
            map_elem = page.query_selector(".folium-map") or page.query_selector(".leaflet-container")
            if map_elem:
                map_elem.screenshot(path=out_path)
            else:
                page.screenshot(path=out_path)
            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Screenshot failed: {e}")

    browser.close()
