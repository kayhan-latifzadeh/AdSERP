import os
import numpy as np
import pandas as pd
import cv2


def draw_ad_boundaries(image, ads_dict, thickness, color):
    for ad_type in ads_dict:
        ads = ads_dict[ad_type]
        if len(ads):
            for ad in ads:
                rec_x, rec_y, rec_w, rec_h = ad['location']['x'], ad[
                    'location']['y'], ad['size']['width'], ad['size']['height']
                cv2.rectangle(image, (rec_x, rec_y), (rec_x + rec_w,
                              rec_y + rec_h), color=color, thickness=thickness)
    return image


def generate_heatmap(coordinators_image, background_image, colormap=cv2.COLORMAP_JET, gaussian_size=(25, 25), gaussian_sigma=8.33):
    gray = cv2.cvtColor(coordinators_image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary_img = cv2.bitwise_not(binary_img)
    blur = cv2.GaussianBlur(binary_img, gaussian_size, gaussian_sigma)
    heatmap_img = cv2.applyColorMap(blur, colormap)
    result = cv2.addWeighted(heatmap_img, 0.8, background_image, 0.2, 0)

    return result


def get_custom_colormap():
    colors = np.array([[0, 255, 0], [51, 255, 0], [102, 255, 0], [153, 255, 0],
                       [204, 255, 0], [255, 255, 0], [
                           255, 204, 0], [255, 153, 0],
                       [255, 102, 0], [255, 51, 0], [255, 0, 0]
                       ])
    colors = colors / 255.0
    color_map = np.array(colors * 255, dtype=np.uint8)
    color_map = cv2.merge((color_map[:, 2], color_map[:, 1], color_map[:, 0]))
    return color_map

def get_color_from_gradient_at(ratio):
    perc = ratio * 100
    r = g = b = 0
    if perc < 50:
        r = 255
        g = round(5.1 * perc)
    else:
        g = 255
        r = round(510 - 5.10 * perc)
    return (b, g, r)

def generate_trajectories(image, xs, ys, thickness=4, colored=False, with_variable_line_thickness=False, min_thickness=1, max_thickness=9):
    color_map = get_custom_colormap()
    for i in range(len(xs)-1):
        x1, y1 = int(xs[i]), int(ys[i])
        x2, y2 = int(xs[i+1]), int(ys[i+1])
        ratio = 1 - (i/len(xs))
        if with_variable_line_thickness:
            
            thickness = int(min_thickness + max_thickness * ratio)

        if colored:
            cv2.line(image, (x1, y1), (x2, y2), get_color_from_gradient_at(ratio=ratio), thickness)
        else:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness)
    return image


def generate_cursor_positions_image(image, xs, ys):
    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)
    return image


def draw_cursor(img, cursor_pos, cursor_size=10, color=(0, 0, 0)):
    cursor_x = cursor_pos[0]
    cursor_y = cursor_pos[1]
    cursor_angle = -130
    cursor_shape = np.array([[-cursor_size, -cursor_size*2],
                            [0, -cursor_size], [cursor_size, -cursor_size*2], [0, 0]])
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), cursor_angle, 1)
    cursor_shape = np.int64(np.dot(rotation_matrix, np.vstack((cursor_shape.T, np.ones(
        (1, cursor_shape.shape[0])))))[0:2].T + np.array([cursor_x, cursor_y]))
    cv2.drawContours(img, [cursor_shape], 0, color, 2)
    cv2.fillPoly(img, [cursor_shape], color)

    return img.copy()