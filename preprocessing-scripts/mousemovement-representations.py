import os
from utils import load_dict_from_file, extract_trial_info, get_files
import numpy as np
import pandas as pd
import cv2
import visualizations

def pad_image_to_height(img, target_height=5094):
    height, width = img.shape[:2]
    if height < target_height:
        padding_needed = target_height - height
        white_padding = np.ones((padding_needed, width, 3), dtype=np.uint8) * 255
        padded_img = np.vstack((img, white_padding))
    else:
        padded_img = img
    return padded_img



mt_dir = 'path/to/mouse-movement-files/directory'
xml_dir = 'path/to/xml-files/directory'
screenshots_dir = 'path/to/screenshots/directory'
ads_boundaries = 'path/to/aoi-info-files/directory'


screenshots = get_files(screenshots_dir)

for screenshot in screenshots:
    if screenshot.endswith('.png'):
        trial_file_id = screenshot.split('.png')[0]

        image = pad_image_to_height(cv2.imread(os.path.join(screenshots_dir, trial_file_id + '.png')))
        image_height, image_width, _ = image.shape
    

        white_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        white_images_mask = visualizations.draw_ad_boundaries(thickness=-1, color=(128, 128, 128), image=white_image.copy(), ads_dict=load_dict_from_file(os.path.join(ads_boundaries, trial_file_id + ".json")))

        display_width = 1280
        display_height = 1024
        win_width = int(extract_trial_info(os.path.join(xml_dir, trial_file_id + ".xml"))['window'].split("x")[0])
        win_height = int(extract_trial_info(os.path.join(xml_dir, trial_file_id + ".xml"))['window'].split("x")[1])

        mt_data_df = pd.read_csv(os.path.join(mt_dir, trial_file_id + '.csv'))
        # extract those rows of data that a mousemove happend
        mt_data_df = mt_data_df[mt_data_df['event'].isin(['mousemove', 'mouseover'])]


        for index, row in mt_data_df.iterrows():
            ratio_x = row['xpos']/win_width
            ratio_y = row['ypos']/win_height
            mt_data_df.loc[index, 'xpos'] = display_width*ratio_x
            mt_data_df.loc[index, 'ypos'] = display_height*ratio_y


        timestamps = mt_data_df['timestamp'].values
        events = mt_data_df['event'].values
        cursor_xs = mt_data_df['xpos'].values
        cursor_ys = mt_data_df['ypos'].values

        position_img = visualizations.generate_cursor_positions_image(xs=cursor_xs, ys=cursor_ys, image=white_images_mask.copy())
        heatmap_image = visualizations.generate_heatmap(coordinators_image=position_img.copy(), background_image=white_image.copy())
        heatmap = cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/heatmaps/without-ads-masks/" + trial_file_id + '.png', heatmap_image)
        heatmap_image_with_ad_maks = visualizations.generate_heatmap(coordinators_image=position_img.copy(), background_image=white_images_mask.copy())
        heatmap = cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/heatmaps/with-ads-masks/" + trial_file_id + '.png', heatmap_image_with_ad_maks)
        
        trajectories_img = visualizations.generate_trajectories(image=white_image, xs=cursor_xs, ys=cursor_ys)
        trajectories_img = visualizations.draw_cursor(img=trajectories_img, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        trajectories_img = visualizations.draw_cursor(img=trajectories_img, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/trajectories/without-ads-masks/" + trial_file_id + '.png', trajectories_img)

        colored_trajectories_img = visualizations.generate_trajectories(image=white_image.copy(), xs=cursor_xs, ys=cursor_ys, colored=True)
        colored_trajectories_img = visualizations.draw_cursor(img=colored_trajectories_img, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        colored_trajectories_img = visualizations.draw_cursor(img=colored_trajectories_img, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/colored-trajectories/without-ads-masks/" + trial_file_id + '.png', colored_trajectories_img)

        trajectories_img_with_ad_maks = visualizations.generate_trajectories(image=white_images_mask.copy(), xs=cursor_xs, ys=cursor_ys)
        trajectories_img_with_ad_maks = visualizations.draw_cursor(img=trajectories_img_with_ad_maks, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        trajectories_img_with_ad_maks = visualizations.draw_cursor(img=trajectories_img_with_ad_maks, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/trajectories/with-ads-masks/" + trial_file_id + '.png', trajectories_img_with_ad_maks)

        colored_trajectories_img_with_ad_maks = visualizations.generate_trajectories(image=white_images_mask.copy(), xs=cursor_xs, ys=cursor_ys, colored=True)
        colored_trajectories_img_with_ad_maks = visualizations.draw_cursor(img=colored_trajectories_img_with_ad_maks, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        colored_trajectories_img_with_ad_maks = visualizations.draw_cursor(img=colored_trajectories_img_with_ad_maks, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/colored-trajectories/with-ads-masks/" + trial_file_id + '.png', colored_trajectories_img_with_ad_maks)

        trajectories_img_with_thickness = visualizations.generate_trajectories(image=white_image, xs=cursor_xs, ys=cursor_ys, with_variable_line_thickness=True)
        trajectories_img_with_thickness = visualizations.draw_cursor(img=trajectories_img_with_thickness, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        trajectories_img_with_thickness = visualizations.draw_cursor(img=trajectories_img_with_thickness, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/trajectories-with-variable-line-thickness/without-ads-masks/" + trial_file_id + '.png', trajectories_img_with_thickness)
        
        trajectories_img_with_thickness_with_masks = visualizations.generate_trajectories(image=white_images_mask, xs=cursor_xs, ys=cursor_ys, with_variable_line_thickness=True)
        trajectories_img_with_thickness_with_masks = visualizations.draw_cursor(img=trajectories_img_with_thickness_with_masks, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        trajectories_img_with_thickness_with_masks = visualizations.draw_cursor(img=trajectories_img_with_thickness_with_masks, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/trajectories-with-variable-line-thickness/with-ads-masks/" + trial_file_id + '.png', trajectories_img_with_thickness_with_masks)

        colored_trajectories_img_with_thickness = visualizations.generate_trajectories(image=white_image, xs=cursor_xs, ys=cursor_ys, colored=True, with_variable_line_thickness=True)
        colored_trajectories_img_with_thickness = visualizations.draw_cursor(img=colored_trajectories_img_with_thickness, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        colored_trajectories_img_with_thickness = visualizations.draw_cursor(img=colored_trajectories_img_with_thickness, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/colored-trajectories-with-variable-line-thickness/without-ads-masks/" + trial_file_id + '.png', colored_trajectories_img_with_thickness)
        
        colored_trajectories_img_with_thickness_with_masks = visualizations.generate_trajectories(image=white_images_mask, xs=cursor_xs, ys=cursor_ys, colored=True, with_variable_line_thickness=True)
        colored_trajectories_img_with_thickness_with_masks = visualizations.draw_cursor(img=colored_trajectories_img_with_thickness_with_masks, cursor_pos=(int(cursor_xs[0]), int(cursor_ys[0])), color=(0, 255, 0))
        colored_trajectories_img_with_thickness_with_masks = visualizations.draw_cursor(img=colored_trajectories_img_with_thickness_with_masks, cursor_pos=(int(cursor_xs[-1]), int(cursor_ys[-1])), color=(0, 0, 255))
        cv2.imwrite("path/to/desired_directory//visualizations/mouse-tracking/colored-trajectories-with-variable-line-thickness/with-ads-masks/" + trial_file_id + '.png', colored_trajectories_img_with_thickness_with_masks)

        print(trial_file_id, 'done!')