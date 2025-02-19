from utils import get_webdriver
from find_native_ads import *
from find_top_ad import *

def get_aoi_boundary(element, zoom_ratio):
    try:
        x, y = int(element.location['x'] * zoom_ratio), int(element.location['y'] * zoom_ratio)
        w, h = int(element.size['width'] * zoom_ratio), int(element.size['height'] * zoom_ratio)
        return x, y, w, h
    except:
        return None

if __name__ == "__main__":
    
    driver = get_webdriver()
    driver.set_window_size(1360, 5000)
    driver.get(f'URL-TO-HTML-FILE.html')
    driver.maximize_window()
    zoom_ratio = 1

    bottom_ads_div, bottom_ads_list = get_bottom_native_ads(driver)
    top_native_ads_div, top_native_ads = get_top_native_ads(driver)
    top_ads_div = get_top_ads_div(driver)
    ldd_ads_div = get_top_carousel_div(top_ads_div)
    rdd_ads_div = get_right_carousel_div(driver)
    
    
    if ldd_ads_div != None:
        x, y, w, h = get_aoi_boundary(ldd_ads_div, zoom_ratio)
        print("left-aligned direct-display ad:", x, y, w, h)
    
    if rdd_ads_div != None:
        x, y, w, h = get_aoi_boundary(rdd_ads_div, zoom_ratio)
        print("right-aligned direct-display ad:", x, y, w, h)

    if bottom_ads_div != None:
        x, y, w, h = get_aoi_boundary(bottom_ads_div, zoom_ratio)
        print("Native ad (top):", x, y, w, h)
    
    if top_native_ads_div != None:
        x, y, w, h = get_aoi_boundary(top_native_ads_div, zoom_ratio)
        print("Native ad (bottom):", x, y, w, h)
    
    driver.close()
  
    
    