import cv2
import torch
import numpy as np


def colormap(rgb=True):
    color_list = np.array(
        [
            0.000, 0.000, 0.000,
            1.000, 1.000, 1.000,
            1.000, 0.498, 0.313,
            0.392, 0.581, 0.929,
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def vis_hoi_add_mask(image, background_mask, person_contour_mask, object_contour_mask, background_color, person_contour_color, object_contour_color, background_alpha, contour_alpha):
    background_color = np.array(background_color)
    person_contour_color = np.array(person_contour_color)
    object_contour_color = np.array(object_contour_color)

    # background_mask = 1 - background_mask
    # contour_mask = 1 - contour_mask

    for i in range(3):
        image[:, :, i] = image[:, :, i] * (1-background_alpha+background_mask*background_alpha) \
            + background_color[i] * (background_alpha-background_mask*background_alpha)
        
        image[:, :, i] = image[:, :, i] * (1-contour_alpha+person_contour_mask*contour_alpha) \
            + person_contour_color[i] * (contour_alpha-person_contour_mask*contour_alpha)

        image[:, :, i] = image[:, :, i] * (1-contour_alpha+object_contour_mask*contour_alpha) \
            + object_contour_color[i] * (contour_alpha-object_contour_mask*contour_alpha)

    return image.astype('uint8')


def vis_add_mask(image, background_mask, contour_mask, background_color, contour_color, background_alpha, contour_alpha):
    background_color = np.array(background_color)
    contour_color = np.array(contour_color)
    background_mask = 1 - background_mask

    for i in range(3):
        image[:, :, i] = image[:, :, i] * (1-background_alpha+background_mask*background_alpha) \
            + background_color[i] * (background_alpha-background_mask*background_alpha)
        
        image[:, :, i] = image[:, :, i] * (1-contour_alpha+contour_mask*contour_alpha) \
            + contour_color[i] * (contour_alpha-contour_mask*contour_alpha)


    return image.astype('uint8')


def hoi_mask_generator_11(person_mask, object_mask, background_radius, contour_radius):
    union_mask = ((person_mask != 0) | (object_mask != 0)).astype(np.uint8)
    dist_transform_fore = cv2.distanceTransform(union_mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1-union_mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back

    background_mask = np.clip(dist_map, -background_radius, background_radius)
    background_mask = (background_mask - np.min(background_mask))
    if np.max(background_mask) != 0:
        background_mask = background_mask / np.max(background_mask)

    person_dist_transform_fore = cv2.distanceTransform(person_mask, cv2.DIST_L2, 3)
    person_dist_transform_back = cv2.distanceTransform(1-person_mask, cv2.DIST_L2, 3)
    person_dist_map = person_dist_transform_fore - person_dist_transform_back

    object_dist_transform_fore = cv2.distanceTransform(object_mask, cv2.DIST_L2, 3)
    object_dist_transform_back = cv2.distanceTransform(1-object_mask, cv2.DIST_L2, 3)
    object_dist_map = object_dist_transform_fore - object_dist_transform_back

    contour_radius += 2
    person_contour_mask = np.abs(np.clip(person_dist_map, -contour_radius, contour_radius))
    person_contour_mask = person_contour_mask / np.max(person_contour_mask)

    object_contour_mask = np.abs(np.clip(object_dist_map, -contour_radius, contour_radius))
    object_contour_mask = object_contour_mask / np.max(object_contour_mask)
    return background_mask, person_contour_mask, object_contour_mask


def mask_generator_11(mask, background_radius, contour_radius):
    dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back

    background_mask = np.clip(dist_map, -background_radius, background_radius)
    background_mask = (background_mask - np.min(background_mask))
    if np.max(background_mask) != 0:
        background_mask = background_mask / np.max(background_mask)

    contour_radius += 2
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask = contour_mask / np.max(contour_mask)

    return background_mask, contour_mask


def mask_hoi_painter(image, person_mask, object_mask, background_alpha=0.5, background_blur_radius=7, contour_width=3, person_contour_color=3, object_contour_color=14, contour_alpha=1, mode='11'):

    w, h, _ = image.shape
    background_radius = (background_blur_radius - 1) // 2
    contour_radius = (contour_width - 1) // 2
    background_mask, person_contour_mask, object_contour_mask = hoi_mask_generator_11(person_mask, object_mask, background_radius, contour_radius)

    painted_image = vis_hoi_add_mask\
        (image, background_mask, person_contour_mask, object_contour_mask, 
        color_list[0], color_list[person_contour_color], color_list[object_contour_color], background_alpha, contour_alpha) # black for background

    return painted_image


def mask_painter(image, mask, background_alpha=0.5, background_blur_radius=7, contour_width=3, contour_color=3, contour_alpha=1, mode='11'):

    w, h, _ = image.shape
    background_radius = (background_blur_radius - 1) // 2
    contour_radius = (contour_width - 1) // 2
    background_mask, contour_mask = mask_generator_11(mask, background_radius, contour_radius)

    painted_image = vis_add_mask\
        (image, background_mask, contour_mask, 
        color_list[contour_color], color_list[1], background_alpha, contour_alpha) # black for background

    return painted_image
