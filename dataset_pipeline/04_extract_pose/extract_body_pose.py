import os
import argparse
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm

from pose_model import PoseModel


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def inference():

    device = torch.device('cuda')

    image_dir = '/data/HOIYouTube/violin/images_temp/0002'
    detection_results = os.path.join('/data/HOIYouTube/violin/bigdetection_temp/0002_30.pkl')
    detection_results = load_pickle(detection_results)

    wholebody_model = PoseModel('ViTPose-H-WholeBody', device)

    count = 0
    for frame_id in tqdm(sorted(list(detection_results.keys()))):
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detection_results[frame_id]
        if detections['person'] is not None:
            bboxes = [{'bbox': box} for box in detections['person']]
            pose_outputs = wholebody_model.predict_pose(image, bboxes)
            kps_outputs = [kps['keypoints'] for kps in pose_outputs]

            if not os.path.exists('./__debug__/{}_wholebody_kps.pkl'.format(frame_id)):
                continue
            pose_outputs = load_pickle('./__debug__/{}_wholebody_kps.pkl'.format(frame_id))
            image = wholebody_model.visualize_pose_results(image, pose_outputs)
            save_pickle('./__debug__/{}_wholebody_kps.pkl'.format(frame_id), pose_outputs)
        # print(kps_outputs)
        cv2.imwrite('./__debug__/{}_kps_vis.jpg'.format(frame_id), image[:, :, ::-1])
        count += 1
        if count > 30:
            exit(0)


def inference_all(args):
    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')
    image_dir = os.path.join(args.root_dir, 'images_temp')
    mask_dir = os.path.join(args.root_dir, 'hoi_mask')

    device = torch.device('cuda')
    wholebody_model = PoseModel('ViTPose-H-WholeBody', device)

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        if not os.path.exists(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id))):
            continue
        tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))

        output_dir = os.path.join(args.root_dir, 'wholebody_kps', video_id)
        os.makedirs(output_dir, exist_ok=True)
        print('Found {} instances.'.format(len(tracking_results['hoi_instances'])))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            # if os.path.exists(os.path.join(output_dir, '{}_wholebody_kps.pkl'.format(hoi_id))):
            #     continue

            wholebody_kps = []
            for item in tqdm(hoi_instance['sequences']):
                frame_id = item['frame_id']
                person_bbox = item['person_bbox']

                if person_bbox is not None:
                    image = cv2.imread(os.path.join(image_dir, video_id, '{}.jpg'.format(frame_id)))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    _person_bbox = np.ones(5, )
                    _person_bbox[:4] = person_bbox
                    bboxes = [{'bbox': _person_bbox}, ]
                    pose_outputs = wholebody_model.predict_pose(image, bboxes)
                    kps = pose_outputs[0]['keypoints']
                else:
                    kps = np.zeros((133, 3))

                wholebody_kps.append({
                    'frame_id': frame_id,
                    'keypoints': kps
                })

            save_pickle(os.path.join(output_dir, '{}_wholebody_kps.pkl'.format(hoi_id)), wholebody_kps)

        print('Video {} done!'.format(video_id))


if __name__ == '__main__':
    # envs: bigdet
    parser = argparse.ArgumentParser('ViTPose inference.')
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )

    args = parser.parse_args()
    # inference()
    inference_all(args)
