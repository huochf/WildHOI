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



def main(args):
    root_dir = args.root_dir
    object_name = root_dir.split('/')[-1]
    with open('/storage/data/huochf/HOIYouTube/train_test_split_{}.json'.format(object_name), 'r') as f:
        train_split = json.load(f)

    frame_intervals = 30 // args.fps
    output_dir = './annotation_contact/{}_test/images'.format(object_name)
    os.makedirs(output_dir, exist_ok=True)

    for video_idx in tqdm(train_split['test']):
        video_id = '{}'.format(video_idx)

        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            sequences = hoi_instance['sequences']

            smplx_params = load_pickle(os.path.join(root_dir, 'smplx_tuned', video_id, '{}_smplx.pkl'.format(hoi_id)))

            sequences = sequences[::frame_intervals]
            smplx_params = smplx_params[::frame_intervals]

            # _output_dir = os.path.join(output_dir, video_id, hoi_id)
            # os.makedirs(os.path.join(_output_dir, 'images'), exist_ok=True)

            for idx, item in enumerate(sequences):
                assert item['frame_id'] == smplx_params[idx]['frame_id']

                cx, cy = smplx_params[idx]['princpt']
                bbox = smplx_params[idx]['bbox'] # xywh
                x1, y1, w, h = bbox
                crop_s = 1.5 * max(h, w)
                image = cv2.imread(os.path.join(root_dir, 'images_temp', video_id, '{}.jpg'.format(item['frame_id'])))
                img_trans = gen_trans_from_patch_cv(cx, cy, crop_s, 1024)
                img_patch = cv2.warpAffine(image, img_trans, (1024, 1024), flags=cv2.INTER_LINEAR)
                image_out = np.ones((1024 + 128 * 2, 1024, 3)) * 255
                image_out[:1024, :1024] = img_patch

                for i in range(2):
                    for j in range(8):
                        cv2.rectangle(image_out, (128 * j, 1024 + 128 * i, ), (128 * j + 128, 1024 + 128 * i + 128), (0, 0, 255), 8)
                        cv2.putText(image_out, str(j), (128 * j + 32, 1024 + 128 * i + 64 + 32), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0, ), 8)

                cv2.imwrite(os.path.join(output_dir, '{}_{}_{}.jpg'.format(video_id, hoi_id, item['frame_id'])), image_out.astype(np.uint8))


def main_v2(args):
    root_dir = args.root_dir
    object_name = root_dir.split('/')[-1]
    with open('/storage/data/huochf/HOIYouTube/train_test_split_{}.json'.format(object_name), 'r') as f:
        train_split = json.load(f)

    output_dir = './annotation_contact_v2/{}/images'.format(object_name)
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(sorted(os.listdir(os.path.join(root_dir, 'object_annotations', 'pose')))):
        video_id, frame_id, hoi_id = file.split('.')[0].split('_')

        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))

        for hoi_instance in tracking_results['hoi_instances']:
            if hoi_instance['hoi_id'] != hoi_id:
                continue
            sequences = hoi_instance['sequences']

            try:
                smplx_params = load_pickle(os.path.join(root_dir, 'smplx_tuned', video_id, '{}_smplx.pkl'.format(hoi_id)))
            except:
                continue

            for idx, item in enumerate(sequences):
                if item['frame_id'] != frame_id:
                    continue
                assert item['frame_id'] == smplx_params[idx]['frame_id']

                cx, cy = smplx_params[idx]['princpt']
                bbox = smplx_params[idx]['bbox'] # xywh
                x1, y1, w, h = bbox
                crop_s = 1.5 * max(h, w)
                image = cv2.imread(os.path.join(root_dir, 'images_temp', video_id, '{}.jpg'.format(item['frame_id'])))
                img_trans = gen_trans_from_patch_cv(cx, cy, crop_s, 1024)
                img_patch = cv2.warpAffine(image, img_trans, (1024, 1024), flags=cv2.INTER_LINEAR)
                image_out = np.ones((1024 + 128 * 2, 1024, 3)) * 255
                image_out[:1024, :1024] = img_patch

                for i in range(2):
                    for j in range(8):
                        cv2.rectangle(image_out, (128 * j, 1024 + 128 * i, ), (128 * j + 128, 1024 + 128 * i + 128), (0, 0, 255), 8)
                        cv2.putText(image_out, str(j + 1), (128 * j + 32, 1024 + 128 * i + 64 + 32), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0, ), 8)

                cv2.imwrite(os.path.join(output_dir, '{}_{}_{}.jpg'.format(video_id, hoi_id, item['frame_id'])), image_out.astype(np.uint8))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--fps', type=int, default=3)
    args = parser.parse_args()
    # main(args)
    main_v2(args)
