import os
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
import argparse

from mmpose.apis import init_model, inference_topdown
from mmpose.visualization.fast_visualizer import FastVisualizer


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def get_bbox_from_kps(kps, threshold):
    x1, y1, _ = kps[kps[:, 2] > threshold].min(axis=0)
    x2, y2, _ = kps[kps[:, 2] > threshold].max(axis=0)
    s = max(y2 - y1, x2 - x1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = max(0, cx - s)
    y1 = max(0, cy - s)
    x2 = cx + s
    y2 = cy + s

    return np.array([x1, y1, x2, y2])


def get_face_bbox(wholebody_kps, threshold=0.5):
    face_kps = wholebody_kps[23:91]

    if np.sum(face_kps[:, 2] > threshold) < 30:
        return None

    bbox = get_bbox_from_kps(face_kps, threshold)

    if (bbox[2] - bbox[0]) < 100:
        return None

    return bbox


def get_hand_bbox(wholebody_kps, is_left=True, threshold=0.5):

    if is_left:
        hand_kps = wholebody_kps[91:112]
    else:
        hand_kps = wholebody_kps[112:133]

    if np.sum(hand_kps[:, 2] > threshold) < 10:
        return None

    bbox = get_bbox_from_kps(hand_kps, threshold)
    # print(hand_kps[:, 2])
    # print(bbox)
    # print(bbox[2] - bbox[0])

    if (bbox[2] - bbox[0]) < 100:
        return None

    return bbox


def extract_face_and_hand():
    device = torch.device('cuda')

    face_model = init_model('./mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py',
                            './pretrained_models/rtmpose-wholebody-face.pth', device)
    hand_model = init_model('./mmpose/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py',
                            './pretrained_models/hrnetv2_w18_wholebody_hand.pth', device)
    
    wholebody_kps = load_pickle('./__debug__/000601_wholebody_kps.pkl')

    image = cv2.imread('/data/HOIYouTube/violin/images_temp/0002/000601.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    for item in wholebody_kps:
        kps = item['keypoints']

        face_bbox = get_face_bbox(kps)
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_image = image[int(y1) : min(h, int(y2)), int(x1) : min(w, int(x2))]
            # cv2.imwrite('./__debug__/000001_face.jpg', face_image_crop[:, :, ::-1])

            face_outputs = inference_topdown(face_model, face_image)
            # print(face_outputs[0].pred_instances.keypoint_scores[0])

            kps[23:91, 0] = face_outputs[0].pred_instances.keypoints[0, :, 0] + x1
            kps[23:91, 1] = face_outputs[0].pred_instances.keypoints[0, :, 1] + y1
            kps[23:91, 2] = face_outputs[0].pred_instances.keypoint_scores[0]

        lhand_bbox = get_hand_bbox(kps, is_left=True)
        if lhand_bbox is not None:
            x1, y1, x2, y2 = lhand_bbox
            lhand_image = image[int(y1) : int(y2), int(x1) : int(x2)]

            left_hand_outputs = inference_topdown(hand_model, lhand_image)

            # print(left_hand_outputs[0].pred_instances.keypoint_scores[0])

            kps[91:112, 0] = left_hand_outputs[0].pred_instances.keypoints[0, :, 0] + x1
            kps[91:112, 1] = left_hand_outputs[0].pred_instances.keypoints[0, :, 1] + y1
            kps[91:112, 2] = left_hand_outputs[0].pred_instances.keypoint_scores[0]

        rhand_bbox = get_hand_bbox(kps, is_left=False)
        if rhand_bbox is not None:
            x1, y1, x2, y2 = rhand_bbox
            rhand_image = image[int(y1) : int(y2), int(x1) : int(x2)]

            right_hand_outputs = inference_topdown(hand_model, rhand_image)

            print(right_hand_outputs[0].pred_instances.keypoint_scores[0])

            kps[112:133, 0] = right_hand_outputs[0].pred_instances.keypoints[0, :, 0] + x1
            kps[112:133, 1] = right_hand_outputs[0].pred_instances.keypoints[0, :, 1] + y1
            kps[112:133, 2] = right_hand_outputs[0].pred_instances.keypoint_scores[0]

    save_pickle('./__debug__/000601_wholebody_kps.pkl', wholebody_kps)


def extract_all(args):
    device = torch.device('cuda')

    face_model = init_model('./mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py',
                            './pretrained_models/rtmpose-wholebody-face.pth', device)
    hand_model = init_model('./mmpose/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py',
                            './pretrained_models/hrnetv2_w18_wholebody_hand.pth', device)

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)

        wholebody_kps_dir = os.path.join(args.root_dir, 'wholebody_kps', video_id)
        image_dir = os.path.join(args.root_dir, 'images_temp', video_id)

        if not os.path.exists(wholebody_kps_dir):
            continue
        print('Find {} sequences.'.format(len(os.listdir(wholebody_kps_dir))))
        for file in os.listdir(wholebody_kps_dir):
            hoi_id = file.split('.')[0].split('_')[0]

            save_dir = os.path.join(args.root_dir, 'wholebody_kps_refined', video_id)
            os.makedirs(save_dir, exist_ok=True)
            # if os.path.exists(os.path.join(save_dir, file)):
            #     continue

            wholebody_kps = load_pickle(os.path.join(wholebody_kps_dir, file))
            for item in tqdm(wholebody_kps):
                frame_id = item['frame_id']
                keypoints = item['keypoints']

                image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape

                face_bbox = get_face_bbox(keypoints)
                if face_bbox is not None:
                    x1, y1, x2, y2 = face_bbox
                    face_image = image[int(y1) : min(h, int(y2)), int(x1) : min(w, int(x2))]

                    face_outputs = inference_topdown(face_model, face_image)

                    keypoints[23:91, 0] = face_outputs[0].pred_instances.keypoints[0, :, 0] + x1
                    keypoints[23:91, 1] = face_outputs[0].pred_instances.keypoints[0, :, 1] + y1
                    keypoints[23:91, 2] = face_outputs[0].pred_instances.keypoint_scores[0]

                left_hand_bbox = get_hand_bbox(keypoints, is_left=True)
                if left_hand_bbox is not None:
                    x1, y1, x2, y2 = left_hand_bbox
                    lhand_image = image[int(y1) : int(y2), int(x1) : int(x2)]

                    left_hand_outputs = inference_topdown(hand_model, lhand_image)

                    keypoints[91:112, 0] = left_hand_outputs[0].pred_instances.keypoints[0, :, 0] + x1
                    keypoints[91:112, 1] = left_hand_outputs[0].pred_instances.keypoints[0, :, 1] + y1
                    keypoints[91:112, 2] = left_hand_outputs[0].pred_instances.keypoint_scores[0]


                right_hand_bbox = get_hand_bbox(keypoints, is_left=False)
                if right_hand_bbox is not None:
                    x1, y1, x2, y2 = right_hand_bbox
                    rhand_image = image[int(y1) : int(y2), int(x1) : int(x2)]

                    right_hand_outputs = inference_topdown(hand_model, rhand_image)

                    keypoints[112:133, 0] = right_hand_outputs[0].pred_instances.keypoints[0, :, 0] + x1
                    keypoints[112:133, 1] = right_hand_outputs[0].pred_instances.keypoints[0, :, 1] + y1
                    keypoints[112:133, 2] = right_hand_outputs[0].pred_instances.keypoint_scores[0]

            save_pickle(os.path.join(save_dir, file), wholebody_kps)

        print('Video {} done.'.format(video_id))


if __name__ == '__main__':
    # envs: mmpose
    parser = argparse.ArgumentParser('ViTPose inference.')
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )

    args = parser.parse_args()
    # extract_face_and_hand()
    extract_all(args)
