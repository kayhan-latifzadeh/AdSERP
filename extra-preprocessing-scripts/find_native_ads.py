from selenium.webdriver.common.by import By


def get_bottom_native_ads(driver):
  """
  Return the div that wrap all the native bottom ads and a list of each native bottom ad
  """
  try:
    bottom_ads_div = driver.find_element(By.ID, value="bottomads")
    bottom_ads_list = bottom_ads_div.find_elements(By.CSS_SELECTOR, value="div[data-text-ad]")
  except:
    return None, None
  else:
    return bottom_ads_div, bottom_ads_list
  

def get_top_native_ads(driver):
  """
  Return the div that wrap all the native top ads and a list of each native top ad
  """
  try:
    top_ads_div = driver.find_element(By.ID, value='tvcap')
    top_native_div = top_ads_div.find_element(By.ID, value='tads')
    top_native_ads = top_native_div.find_elements(By.CSS_SELECTOR, value="div[data-text-ad]")
  except:
    return None, None
  else:
    return top_native_div, top_native_ads
  

def get_native_ad_header(native_ad):
  """
  Return the header of a native ad (can top or bottom native ad)
  """
  header = native_ad.find_element(By.CSS_SELECTOR, value="div[role='heading']")
  
  return header


def get_native_ad_link(header):
  """
  Return the link of the ad based on its header
  """
  anchor = header.find_element(By.XPATH, value='parent::*')
  link = anchor.find_element(By.XPATH, value='parent::*')
  
  return link


def get_native_ad_text(link):
  """
  Return the text of the ad based on its link
  """
  text = link.find_element(By.XPATH, value='following-sibling::*')
  
  return text

def get_nested_native_ad(native_ad):
  """
  Return nested native ad from a native ad
  """
  try:
    nested_ad = native_ad.find_element(By.CSS_SELECTOR, value="div[role='list']")
  except:
    return None
  else:
    return nested_ad

def get_nested_ad_headers(nested_ad):
  nested_ad_headers = nested_ad.find_elements(By.CSS_SELECTOR, value="div[role='listitem']")
  
  return nested_ad_headers

def get_nested_ad_text(nested_ad_header):
  temp = nested_ad_header.find_element(By.XPATH, value='parent::*')
  nested_ad_text = temp.find_element(By.XPATH, value='following-sibling::*')
  
  return nested_ad_text

def get_native_rating(native_ad):
  try:
    rating = native_ad.find_element(By.TAG_NAME, value='w-ad-seller-rating')
  except:
    return None
  else:
    return rating
  


  
