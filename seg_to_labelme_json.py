import os
import numpy as np
import cv2
import glob
from operator import itemgetter
from tqdm import tqdm

from labelme_json import LabelmeJSON, LabelmeJSONShape


BASE_DIR = os.path.join('.', 'data')
ORG_DIR = os.path.join(BASE_DIR, 'images')
MSK_DIR = os.path.join(BASE_DIR, 'masks')
IMAGE_EXTENTAIONS = ['.png', '.jpg', '.jpeg']
MASK_BINARY_THRESHOLD = 1

NONE_ID = -1
USE_CHAIN_APPROX = cv2.CHAIN_APPROX_TC89_KCOS
#USE_CHAIN_APPROX = cv2.CHAIN_APPROX_NONE
#USE_CHAIN_APPROX = cv2.CHAIN_APPROX_SIMPLE
#USE_CHAIN_APPROX = cv2.CHAIN_APPROX_TC89_L1

ORG_IMG_NAME_INDEX = 0
LABEL_NAME_INDEX = 1


def get_first_contor_id(hierarchy):
    first_contor_index = 0

    for k ,h in enumerate(hierarchy):
        _, prev_id, _, pparent_id = h
        if prev_id == NONE_ID and pparent_id ==  NONE_ID:
            first_contor_index = k
            break

    return first_contor_index


def follow_contor(contor_id, hierarchy, contours, disp_prefix='- ', is_inside=False):
    contor_info = hierarchy[contor_id]
    next_id, prev_id, child_id, parent_id = contor_info

    #print(disp_prefix, ' %03d : ' % contor_id, contor_info)

    contour = contours[contor_id]
    contour = np.squeeze(contour, axis=1)
    cnts = []
    child_ids = []
    child_contours = []
    if child_id != NONE_ID:
        child_ids, child_contours, child_cnts = follow_contor(child_id, hierarchy, contours
                                                              , disp_prefix[:-2] + '     - '
                                                              , is_inside=not is_inside)
        cnts += child_cnts

    next_ids = []
    next_contours = []
    if next_id != NONE_ID:
        next_ids, next_contours, next_cnts = follow_contor(next_id, hierarchy, contours, disp_prefix)
        cnts += next_cnts

    if not is_inside:
        cnt = insert_child_contours(contour, child_contours)
        cnts.append(cnt.tolist())

    same_tier_ids = [contor_id] + next_ids
    same_tier_contours = [contour] + next_contours
    return same_tier_ids, same_tier_contours, cnts


def insert_child_contours(contour, child_contours):
    insert_indexes = []
    start_indexes = []
    for child_contour in child_contours:
        if len(child_contour) < 1:
            continue
        insert_index, start_index = get_closest_points_indexes(contour, child_contour)
        insert_indexes.append(insert_index)
        start_indexes.append(start_index)

    child_cnts_with_ins_idx = list(zip(insert_indexes, start_indexes, child_contours))
    child_cnts_with_ins_idx.sort(key=itemgetter(0))
    child_cnts_with_ins_idx.reverse()
    cnts = contour.copy()
    for ins_idx, st_idx, child_cnt in child_cnts_with_ins_idx:
        ins_cnt = np.concatenate([child_cnt[st_idx:], child_cnt[:st_idx]])
        cnts = np.concatenate([cnts[:ins_idx+1], ins_cnt, ins_cnt[0:1], cnts[ins_idx:]])
    return cnts


def get_closest_points_indexes(contour_a, contour_b):
    a_count = len(contour_a)
    b_count = len(contour_b)
    tile_a = np.concatenate([contour_a]*len(contour_b))
    tile_b = np.tile(contour_b, len(contour_a))
    tile_b = np.reshape(tile_b, np.shape(tile_a))
    distances = get_distances(tile_a, tile_b)
    ins_idx = np.argmin(distances)
    a_idx = ins_idx % a_count
    b_idx = ins_idx // a_count
    return a_idx, b_idx


def get_distances(arr_a, arr_b):
    diff_x = arr_a[:, 0] - arr_b[:, 0]
    diff_y = arr_a[:, 1] - arr_b[:, 1]
    diff_x *= diff_x
    diff_y *= diff_y
    return np.sqrt(diff_x + diff_y)


def get_points_from_mask_img(mask_file):
    mask_img = cv2.imread(mask_file)
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_img_gray = binary_gray_image(mask_img_gray, MASK_BINARY_THRESHOLD)
    _, contours, hierarchy = cv2.findContours(mask_img_gray, cv2.RETR_CCOMP, USE_CHAIN_APPROX)
    if hierarchy is None:
        return []
    if len(hierarchy) < 1:
        return []
    contor_id = get_first_contor_id(hierarchy[0])
    _, _, masks = follow_contor(contor_id, hierarchy[0], contours)
    return masks


def binary_gray_image(img, th):
    img[img < th] = 0
    img[img != 0] = 255
    return img


def get_image_ext_pathes(base_path):
    pathes = []
    for ext in IMAGE_EXTENTAIONS:
        pathes += glob.glob(base_path + ext)
    return pathes


def save_shape_to_json(original_name, shapes):
    org_img_pathes = get_image_ext_pathes(os.path.join(ORG_DIR, original_name))
    created_count = 0
    if len(org_img_pathes) > 0:
        labelme_json = LabelmeJSON(org_img_pathes[0], shapes=shapes)
        labelme_json.dump()
        created_count = 1
    return created_count


def execute():
    mask_files = get_image_ext_pathes(os.path.join(MSK_DIR, '*'))
    mask_files.sort()

    original_name = ''
    shapes = []
    created_json_count = 0

    pbar = tqdm(total=len(mask_files), desc="Transform mask image", unit=" files")
    for mask_file in mask_files:
        basename = os.path.basename(mask_file)
        name_parts = basename.split('_')

        if len(name_parts) < 3:
            continue

        if original_name == '':
            original_name = name_parts[ORG_IMG_NAME_INDEX]

        if original_name != name_parts[ORG_IMG_NAME_INDEX]:
            created_json_count += save_shape_to_json(original_name, shapes)
            original_name = name_parts[ORG_IMG_NAME_INDEX]
            shapes = []
        points_list = get_points_from_mask_img(mask_file)
        for points in points_list:
            shapes.append(LabelmeJSONShape(name_parts[LABEL_NAME_INDEX], points=points))
        pbar.update(1)
    created_json_count += save_shape_to_json(original_name, shapes)
    pbar.close()

    print('')
    print('### Created JSON : ', created_json_count, ' files')




if __name__ == '__main__':
    execute()
