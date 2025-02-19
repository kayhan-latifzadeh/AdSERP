import os
import xml.etree.ElementTree as ET
import cv2
from selenium import webdriver
import json


def get_files(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


def get_dirs(dir):
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

def extract_trial_info(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    task = root.findall("task")
    batch = task[0].text.split(" | ")[0].strip().split("_")[1]
    trial_number = task[0].text.split(" | ")[0].strip().split("_")[3]
    slug = task[0].text.split(" | ")[1].strip()

    info_dict = {'date': root.find("date").text,
                 'user-agent': root.find("ua").text,
                 'batch': int(batch),
                 'trial-number': int(trial_number),
                 'slug': slug,
                 'screen': root.find("screen").text,
                 'window': root.find("window").text,
                 'document': root.find("document").text,
                 }

    return info_dict

def get_webdriver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    return driver

def load_dict_from_file(file_path):
    with open(file_path, "r") as f:
        dict_obj = json.load(f)
    return dict_obj


if __name__ == '__main__':
    print('hello world!')
