import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def gen_trans_from_patch_cv(box_center_x, box_center_y, box_size, out_size):
    src_w = src_h = box_size

    src_center = np.array([box_center_x, box_center_y], dtype=np.float32)
    src_rightdir = src_center + np.array([0, src_w * 0.5])
    src_downdir = src_center + np.array([src_h * 0.5, 0])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_rightdir
    src[2, :] = src_downdir

    dst = np.array([[out_size / 2, out_size / 2], [out_size / 2, out_size], [out_size, out_size / 2]], dtype=np.float32)
    trans = cv2.getAffineTransform(src, dst)
    return trans


drop_images = {
    'violin': [], #['0463_001_001754', ],
    'basketball': [],
}


def main(args):
    root_dir = args.root_dir
    object_name = args.root_dir.split('/')[-1]

    hoi_recon_results = load_pickle(os.path.join('hoi_recon_with_contact', '{}_test.pkl'.format(object_name)))
    _output_dir = os.path.join('annotation_RT', object_name)

    for item in tqdm(hoi_recon_results):
        image_id = item['image_id']
        # if image_id in drop_images[object_name]:
        #     continue
        video_id, hoi_id, frame_id = image_id.split('_')

        os.makedirs(os.path.join(_output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(_output_dir, 'params'), exist_ok=True)

        cam_R = item['hoi_rotmat']
        cam_T = item['hoi_trans']

        fx, fy = item['focal']
        cx, cy = item['princpt']
        bbox = item['crop_bboxes'] # xywh
        x1, y1, w, h = bbox
        crop_s = 1.5 * max(h, w)
        image = cv2.imread(os.path.join(root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
        img_trans = gen_trans_from_patch_cv(cx, cy, crop_s, 1024)
        img_patch = cv2.warpAffine(image, img_trans, (1024, 1024), flags=cv2.INTER_LINEAR)

        focal = np.array([fx / crop_s * 1024, fy / crop_s * 1024])
        princpt = np.array([512, 512])

        save_dict = {
            'cam_R': cam_R, 
            'cam_T': cam_T, 
            'focal': focal,
            'princpt': princpt,
            'betas': item['smplx_betas'],
            'body_pose': item['smplx_body_pose'],
            'lhand_pose': item['smplx_lhand_pose'],
            'rhand_pose': item['smplx_rhand_pose'],
            'object_rel_rotmat': item['obj_rel_rotmat'],
            'object_rel_trans': item['obj_rel_trans'],
            'object_scale': item['object_scale'],}
        save_pickle(os.path.join(_output_dir, 'params', '{}.pkl'.format(item['image_id'])), save_dict)
        cv2.imwrite(os.path.join(_output_dir, 'images', '{}.jpg'.format(item['image_id'])), img_patch)


def update_annotations(args):
    root_dir = args.root_dir
    object_name = args.root_dir.split('/')[-1]

    hoi_recon_results = load_pickle(os.path.join('hoi_recon_with_contact', '{}_test.pkl'.format(object_name)))
    hoi_recon_results = {item['image_id']: item for item in hoi_recon_results}
    _output_dir = os.path.join('annotation_hoi', object_name, 'test')
    os.makedirs(_output_dir, exist_ok=True)

    for file in os.listdir(os.path.join('./annotation_RT', object_name, 'params')):
        img_id = file.split('.')[0]
        annotation_RT = load_pickle(os.path.join('./annotation_RT', object_name, 'params', file))
        annotation_contact = hoi_recon_results[img_id]

        save_dict = {
            'smplx_betas': annotation_contact['smplx_betas'],
            'smplx_body_pose': annotation_contact['smplx_body_pose'],
            'smplx_lhand_pose': annotation_contact['smplx_lhand_pose'],
            'smplx_rhand_pose': annotation_contact['smplx_rhand_pose'],
            'obj_rel_rotmat': annotation_RT['object_rel_rotmat'],
            'obj_rel_trans': annotation_RT['object_rel_trans'],
            'hoi_rotmat': annotation_contact['hoi_rotmat'],
            'hoi_trans': annotation_contact['hoi_trans'],
            'object_scale': annotation_RT['object_scale'],
            'crop_bboxes': annotation_contact['crop_bboxes'],
            'focal': annotation_contact['focal'],
            'princpt': annotation_contact['princpt'],
        }
        save_pickle(os.path.join(_output_dir, '{}.pkl'.format(img_id)), save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,)
    args = parser.parse_args()
    main(args)
    # update_annotations(args)
