# got from SAMAug : https://github.com/yhydhx/SAMAug

import numpy as np
import cv2
import sys 
# append current directory to sys.path
sys.path.append('./')

from vst_main.Testing import VST_test_once

"""Random Sample Point"""
# def get_random_point(mask):
#   indices = np.argwhere(mask==True)

#   random_point = indices[np.random.choice(list(range(len(indices))))]
#   random_point = [random_point[1], random_point[0]]
#   return random_point

def get_random_point(mask):
    indices = np.argwhere(mask)
    if len(indices) == 0:
        # Fallback: return center or random point in the mask shape
        h, w = mask.shape[:2]
        return [w // 2, h // 2]
    random_point = indices[np.random.choice(len(indices))]
    return [random_point[1], random_point[0]]


"""Max Entropy Point"""
def image_entropy(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return entropy


def calculate_image_entroph(img1, img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    # print(img2)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    # print("Entropy Difference:", entropy_diff)
    return entropy_diff


def select_grid(image, center_point, grid_size):
    (img_h, img_w, _) = image.shape

    # Extract the coordinates of the center point
    x, y = center_point
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2) if x - (grid_size // 2) > 0 else 0
    top_left_y = y - (grid_size // 2) if y - (grid_size // 2) > 0 else 0
    bottom_right_x = top_left_x + grid_size if top_left_x + grid_size < img_w else img_w
    bottom_right_y = top_left_y + grid_size if top_left_y + grid_size < img_h else img_h

    # Extract the grid from the image
    grid = image[top_left_y: bottom_right_y, top_left_x: bottom_right_x]

    return grid


def get_entropy_points(input_point,mask,image):
    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image, input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            max_entropy = entropy_diff
    return [max_entropy_point[1], max_entropy_point[0]]


"""Max Distance Point"""
def get_distance_points(input_point, mask):
    max_distance_point = [0,0]
    max_distance = 0
    # grid_size = 9
    # center_grid = select_grid(image,input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        distance = np.sqrt((x- input_point[0])**2 + (y- input_point[1]) ** 2)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance
    return [max_distance_point[1],max_distance_point[0]]


"""Saliency Point"""
def get_saliency_point(img, mask, img_name=None, save_img_path=None):
    (img_h, img_w, _) = img.shape

    coor = np.argwhere(mask > 0)
    ymin = min(coor[:, 0])
    ymax = max(coor[:, 0])
    xmin = min(coor[:, 1])
    xmax = max(coor[:, 1])

    xmin2 = xmin - 10 if xmin - 10 > 0 else 0
    xmax2 = img_w if xmax + 10 > img_w else xmax + 10
    ymin2 = ymin - 10 if ymin - 10 > 0 else 0
    ymax2 = img_h if ymax + 10 > img_h else ymax + 10

    vst_input_img = img[ymin2:ymax2, xmin2:xmax2, :]

    # VST mask
    vst_mask = VST_test_once(img_path=vst_input_img)

    # judge point in the vst mask
    vst_indices = np.argwhere(vst_mask > 0)
    random_index = np.random.choice(len(vst_indices), 1)[0]
    # vst_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]
    vst_roi_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]

    # plt.imshow(vst_input_img)
    # plt.axis('off')
    # show_mask(np.array(vst_mask > 0).astype(int), plt.gca())
    # show_points(np.array([vst_roi_random_point]), np.array([1]), plt.gca())
    # plt.savefig(osp.join(save_img_path,
    #                      "{}_5_vst_mask_point.jpg".format(img_name.split('.')[0])), bbox_inches='tight', dpi=100,
    #             pad_inches=0)
    # plt.clf()

    vst_random_point = [vst_roi_random_point[0] + xmin - 10, vst_roi_random_point[1] + ymin - 10]

    return vst_random_point