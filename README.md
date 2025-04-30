# AdSERP
A Versatile Dataset of Mouse and Eye Movements on Search Engine Results Pages

<img src="https://github.com/kayhan-latifzadeh/AdSERP/blob/main/logo.png?raw=true" alt="AdSERP Logo" width="300">

## Data file naming convention

Please download the dataset from [here](https://zenodo.org/records/15236546). Each type of data (e.g. eye movement, mouse movement etc) has this naming convention:  

`[PARTICPANT_ID]-b[BLOCK_NUMBER]-t[TRIAL_NUMBER]`

Example: `p004-b1-t3`

If you are looking for the corresponding data of a participant, for example `p004` in block number 1 and in trial number 3, then the corresponding HTML file is in `data/serp/p004-b1-t3.html`, the fixation file is in `data/fixation-data/p004-b1-t3.csv`, the AOIs of the ads are in `data/ad-boundary-data/p004-b1-t3.json` and so on.

## Data Description

### Participant’s demographics

The (comma-delimited) `data/participants.csv` file has 8 columns:

  - `participant_id`: (string) Participant's ID.
  - `gender`: (string) Participant's gender; ['Female', 'Male', 'Prefer not to say'].
  - `age`: (integer) Participant's age.
  - `english`: (string) Participant's English level; ['Intermediate', 'Advanced', 'Native speaker'].
  - `education`:  (string) Participant's education level; ['High school diploma', "Bachelor's degree", 'Graduate'].
  - `pref_device`: (string) Participant's preferred device for online shopping; ['Smartphones', 'Computers or laptops'].
  - `pref_website`: (string) Preferred website for online shopping; ['Amazon', 'eBay', 'Alibaba', 'Other'].
  - `online_shopping_freq`: (string) Participant's online shopping frequency; ['Rarely', 'Weekly', 'Monthly'].

  Example:
  ```csv
  participant_id,gender,age,english,education,pref_device,pref_website,online_shopping_freq
  p005,Female,36,Intermediate,Graduate,Computers or laptops,Rarely,Amazon
  ...
  ```

### Stimulus pages (HTML files)

The `data/serps/` folder contains page snapshots (in HTML format) of the search engine results.

### Stimulus pages (full-page screenshots)

Rendered full-page screenshots of the corresponding SERPs are in `data/screen-recordings/`. All screenshots have the same viewport width (1280 px) but vary in height, depending on the document size. 

Example:

<img src="https://github.com/kayhan-latifzadeh/AdSERP/blob/main/full-screenshot-example.png?raw=true" alt="Full-page screenshots example" width="300">

### Ad boundary files
These files in `data/ad-boundary-data` contain the slot boundaries of organic and direct-display ads on the screenshots in [JSON](https://www.json.org/json-en.html) format.

  Each file contains 3 entries (ad types):

  - `native_ad`: Refers to organic (or native) ads
  - `dd_top`: Refers to direct display ads that appear on the top-left of the SERP
  - `dd_right`: Refers to direct display ads that appear on the right side of the SERP

  For each ad type, the boundaries are defined with a list of dictionary-like items that contain:
  - `location`: `x` and `y` denote the top-corner coordinates of the AOI (relative to the top-left corner of each screenshot in pixels)
  - `size`: `width` and `height` denote the width and height of the ad, respectively, in pixels.

  **Note:** For an ad type if there is no instance on the corresponding SERP, the value is an empty list `[]`.

```json
{
  "native_ad": [
    {
      "location": {
        "x": 162,
        "y": 1900
      },
      "size": {
        "height": 129,
        "width": 540
      }
    },
    {
      "location": {
        "x": 162,
        "y": 2056
      },
      "size": {
        "height": 108,
        "width": 540
      }
    },
    {
      "location": {
        "x": 162,
        "y": 2191
      },
      "size": {
        "height": 108,
        "width": 540
      }
    }
  ],
  "dd_top": [
    {
      "location": {
        "x": 162,
        "y": 158
      },
      "size": {
        "height": 351,
        "width": 586
      }
    }
  ],
  "dd_right": []
}

```
  An example of how the AOI information can be loaded as a Python dictionary:
```python
import json
import pprint

def load_dict_from_file(file_path):
    with open(file_path, "r") as f:
        dict_obj = json.load(f)
    return dict_obj

# example usage:
trial_ads_boundaries_dict = load_dict_from_file('path_to_the_ad_boundary_json_file')
pprint.pprint(trial_ads_boundaries_dict)
```
### Eye-tracking data

The (comma-delimited) pupil dilation files in `data/pupil-data` has 7 columns:

  - `timestamp`: (int) Unix timestamps in milliseconds
  - `BPOGX`: (int) Rhe best position of gaze x-coordinate, relative to the top-left corner of the screenshot in pixels
  - `BPOGX`: (int) Rhe best position of gaze y-coordinate, relative to the top-left corner of the screenshot in pixels
  - `LPD`: (int) Left eye pupil dilation in pixels
  - `LPV`: (int) The valid flag of the left pupil with the value of 1 if the data is valid, and 0 if it is not
  - `RPD`: (int) Right eye pupil dilation in pixels
  - `RPV`: (int) The valid flag of the right pupil with the value of 1 if the data is valid, and 0 if it is not

  Example:
  ```csv
  timestamp,BPOGX,BPOGX,LPD,LPV,RPD,RPV
  1671196599208,676,204,15.85818,1,14.75008,1,1065,463
  ...
  ```

The (comma-delimited) eye fixation files in `data/fixation-data` has 4 columns:

  - `timestamp`: (int) Unix timestamps in milliseconds
  - `FPOGX`: (int)  Fixation positions of X, relative to the top-left corner of the screenshot in pixels
  - `FPOGY`: (int) Fixation positions of Y, relative to the top-left corner of the screenshot in pixels
  - `FPOGD`: (int) Fixation duration in milliseconds

  Example:
  ```csv
  timestamp,FPOGX,FPOGY,FPOGD
  1671196599208,676,198,476
  ...
  ```

### Mouse tracking data

The `data/mouse-movement-data` folder has all the log files, as recorded by the [evtrack](https://github.com/luileito/evtrack) software.

  Here you can find space-delimited files (CSV) with information about each event type:

  Each CSV file has 8 columns:
  * `timestamp`: (int) Timestamp of the event, with millisecond precision.
  * `xpos`: (int) X position of the mouse cursor.
  * `ypos`: (int) Y position of the mouse cursor.
  * `event`: (string) Browser's event name; e.g. `load`, `mousemove`, `click`, etc.
  * `xpath` (string) Target element that relates to the event, [in XPath notation](https://en.wikipedia.org/wiki/XPath).

  ```csv
  timestamp,xpos,ypos,event,xpath
  1671196599208,0,0,load,/
  ...
  ```

  The XML files in `data/trial-metadata` contain information about the user’s browser metadata (e.g. viewport size, user agent string).

  Example of the XML files:
  ```xml
    <data>
     <ip>::1</ip>
     <date>12/16/2022 14:16:39</date>
     <timestamp>1671196599</timestamp>
     <url>http://localhost/page.php?q=buy-momentum-brands-9-led-flashlight</url>
     <ua>Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36</ua>
     <screen>1280x1024</screen>
     <window>1422x1137</window>
     <document>1403x2642</document>
     <task>batch_20_trial_1 | buy-momentum-brands-9-led-flashlight</task>
    </data>
  ```

How to: (1) load the metadata from XML file as a dictionary, (2) load mouse movement data and make the coordinate relative to the top-corner of the screenshots:

```python
import pandas as pd
import xml.etree.ElementTree as ET

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


mt_data_df = pd.read_csv('path_to_mouse_movement_csv_file')
trial_info_dict = extract_trial_info('path_to_corresponding_xml_file')


# screen size of the monitor
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 1024

# window size of the web browser
win_width = trial_info_dict['window-width']
win_height = trial_info_dict['window-height']

# extract those rows of data that a mousemove happend
mt_data_df = mt_data_df[mt_data_df['event'].isin(['mousemove', 'mouseover'])]

# converting the xs, ys to absolute positions on the corresponding screenshot
for index, row in mt_data_df.iterrows():
    ratio_x = row['xpos']/win_width
    ratio_y = row['ypos']/win_height
    mt_data_df.loc[index, 'xpos'] = int(DISPLAY_WIDTH*ratio_x)
    mt_data_df.loc[index, 'ypos'] = int(DISPLAY_HEIGHT*ratio_y)

# let's get each column as a list of values
timestamps = mt_data_df['timestamp'].values
events = mt_data_df['event'].values
cursor_xs = mt_data_df['xpos'].values
cursor_ys = mt_data_df['ypos'].values
xpaths = mt_data_df['xpath'].values
```

### Screen recordings

In `data/screen-recordings` you can find screen recordings for each trial.

Example (sped up and converted a GIF):

<img src="https://github.com/kayhan-latifzadeh/AdSERP/blob/main/trial-recording-example.gif?raw=true" alt="AdSERP Trial Recording Example" width="500">

## Citation

If you use the dataset, please cite us using this bibliographic reference:

* Kayhan Latifzadeh, Jacek Gwizdka, Luis A. Leiva. 2025. **A Versatile Dataset of Mouse and Eye Movements on Search Engine Results Pages**. <em>In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval</em>. https://doi.org/10.1145/3726302.3730325

```bibtex
@inproceedings{latifzadeh2025adserp,
  title={A Versatile Dataset of Mouse and Eye Movements on Search Engine Results Pages},
  author={Latifzadeh, Kayhan and Gwizdka, Jacek and Leiva, Luis A.},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025},
  doi={10.1145/3726302.3730325}
}
```

## Related papers

The AdSERP dataset has been used in the paper(s) listed below. If you use the dataset in your publications, please email us, and we will include your paper here.

* Mario Villaizán-Vallelado, Matteo Salvatori, Kayhan Latifzadeh, Antonio Penta, Luis A. Leiva, Ioannis Arapakis. 2025. **AdSight: Scalable and Accurate Quantification of User Attention in Multi-Slot Sponsored Search**. <em>In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval</em>. https://doi.org/10.1145/3726302.3729891
