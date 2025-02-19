from selenium.webdriver.common.by import By

def get_top_ads_div(driver):
  try:
    top_ads_div = driver.find_element(By.ID, value='tvcap')
  except:
    return None
  else:
    return top_ads_div


def get_top_carousel_div(top_ads_div):
  """
  Return the div containing the carousel at the top of the page
  """
  try:
    top_carousel_div = top_ads_div.find_element(By.CLASS_NAME, value='commercial-unit-desktop-top')
  except:
    return None
  else:
    return top_carousel_div
  

def get_right_carousel_div(driver):
  """
  Return the div containing the carousel at the right of the page
  """
  try:
    right_carousel_div = driver.find_element(By.CLASS_NAME, value='commercial-unit-desktop-rhs')
  except:
    return None
  else:
    return right_carousel_div
  

def get_top_carousel_header(top_carousel_div):
  """
  Return the header of the top carousel from the div element
  """
  carousel_div_header = top_carousel_div.find_element(By.CSS_SELECTOR, value="h3[role='heading']")

  return carousel_div_header
def get_right_carousel_header(right_carousel_div):
  """
  Return the header of the right carousel from the div element
  """
  carousel_div_header = right_carousel_div.find_element(By.CSS_SELECTOR, value="div[role='heading']")

  return carousel_div_header
  
  

def get_carousel_ads(carousel_div):
  """
  Return a list of all ads inside a carousel from the carousel div (can be top or right carousel)
  """
  temp = carousel_div.find_elements(By.CLASS_NAME, value='pla-unit')
  carousel_ads = []
  #Get only the ads shown on the screenshot.
  for ad in temp:
    ad_x = ad.location['x']
    carousel_x = carousel_div.location['x']
    carousel_w = carousel_div.size['width']
   
    if ad_x <= carousel_x + carousel_w:
      carousel_ads.append(ad)
  return carousel_ads

def get_carousel_ad_image(carousel_div):
  """
  Return a list of all ads's images from a carousel div (can be top or right carousel)
  """
  temp = carousel_div.find_elements(By.CLASS_NAME, value='pla-unit-img-container-link')
  carousel_imgs = []
  #Get only the imgs shown on the screenshot.
  for img in temp:
      img_x = img.location['x']
      carousel_x = carousel_div.location['x']
      carousel_w = carousel_div.size['width']
    
      if img_x <= carousel_x + carousel_w:
        carousel_imgs.append(img)
  return carousel_imgs

def get_carousel_ad_info(carousel_image):
  """
  Return each ad's info from a list of carousel image
  
  ads's info includes everything except the ad's image
  """
  carousel_image_div = carousel_image.find_element(By.XPATH, value='parent::*')
  carousel_ad_info = carousel_image_div.find_element(By.XPATH, value='following-sibling::*')
  
  return carousel_ad_info
  
  
def get_info_title(carousel_ad_info):
  """
  Return the title of an individual ad from an individual ad info
  """
  info_title = carousel_ad_info.find_element(By.CLASS_NAME, value='pla-unit-title')
  
  return info_title

def get_info_price(info_title):
  """
  Return the price of an individual ad from an individual ad title
  """
  info_price = info_title.find_element(By.XPATH, value='following-sibling::*')
  
  return info_price

def get_top_or_right_rating(carousel_ad_info):
  try:
    rating = carousel_ad_info.find_element(By.CSS_SELECTOR, value='span[aria-label^="Rated"]')
  except:
    return None
  else:
    return rating.find_element(By.XPATH, value='./../../..')
  

  