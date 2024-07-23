import os
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry, SamPredictor
from tracker.base_tracker import BaseTracker


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def extract_bbox_from_mask(mask):
    try:
        indices = np.array(np.nonzero(np.array(mask)))
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return None


def box_iou(box1, box2):
    union_x1y1 = np.minimum(box1[:2], box2[:2])
    union_x2y2 = np.maximum(box1[2:4], box2[2:4])

    inter_x1y1 = np.maximum(box1[:2], box2[:2])
    inter_x2y2 = np.minimum(box1[2:4], box2[2:4])

    union_area = (union_x2y2 - union_x1y1).prod()
    inter_w, inter_h = (inter_x2y2 - inter_x1y1)
    inter_area = max(0, inter_w) * max(0, inter_h)
    iou = inter_area / ((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area)
    return iou


def save_mask(person_mask, object_mask, out_path):
    save_dict = {'human': {'mask': None, 'mask_box': None, 'mask_shape': None}, 
                 'object': {'mask': None, 'mask_box': None, 'mask_shape': None}}

    if person_mask is not None and not np.sum(person_mask) == 0:
        ys, xs = np.nonzero(person_mask)
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        person_mask = person_mask[y1:y2+1, x1:x2+1]
        save_dict['human']['mask'] = np.packbits(person_mask)
        save_dict['human']['mask_box'] = np.array([x1, y1, x2, y2])
        save_dict['human']['mask_shape'] = person_mask.shape
    if object_mask is not None and not np.sum(object_mask) == 0:
        ys, xs = np.nonzero(object_mask)
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        object_mask = object_mask[y1:y2+1, x1:x2+1]
        save_dict['object']['mask'] = np.packbits(object_mask)
        save_dict['object']['mask_box'] = np.array([x1, y1, x2, y2])
        save_dict['object']['mask_shape'] = object_mask.shape

    with open(out_path, 'wb') as f:
        pickle.dump(save_dict, f)


def segment_and_track(args):
    device = torch.device('cuda')

    image_dir = os.path.join(args.root_dir, 'images_temp')

    sam_model = sam_model_registry['vit_h'](checkpoint='./weights/sam_vit_h_4b8939.pth')
    sam_predictor = SamPredictor(sam_model.to(device))
    object_tracker = BaseTracker('./weights/XMem-s012.pth', device=device)
    human_tracker = BaseTracker('./weights/XMem-s012.pth', device=device)

    fine_tracking_dir = os.path.join(args.root_dir, 'hoi_tracking')
    os.makedirs(fine_tracking_dir, exist_ok=True)

    for video_idx in range(args.begin_idx, args.end_idx):
        coarse_tracking_results = load_pickle(os.path.join(args.root_dir, 'bigdetection_temp', '{:04d}_track.pkl'.format(video_idx)))

        fine_tracking_results = {'hoi_instances': []}

        # if os.path.exists(os.path.join(fine_tracking_dir, '{:04d}_tracking.pkl'.format(video_idx))):
        #     print('Find file {}'.format(os.path.join(fine_tracking_dir, '{:04d}_tracking.pkl'.format(video_idx))))
        #     continue
        print('Find {} HOI instances.'.format(len(coarse_tracking_results['hoi_instances'])))
        for hoi_instance in coarse_tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            bboxes = hoi_instance['boxes']

            fine_instance = {'hoi_id': hoi_id, 'sequences': []}

            all_frames = sorted(bboxes.keys())
            frame_begin_idx = int(all_frames[0])
            frame_end_idx = int(all_frames[-1])

            save_dir = os.path.join(args.root_dir, 'hoi_mask', '{:04d}'.format(video_idx), '{}'.format(hoi_id))
            os.makedirs(save_dir, exist_ok=True)

            human_in_tracking = False
            object_in_tracking = False
            human_tracker.clear_memory()
            object_tracker.clear_memory()
            for frame_idx in tqdm(range(frame_begin_idx, frame_end_idx)):
                frame_id = '{:06d}'.format(frame_idx)

                image = cv2.imread(os.path.join(image_dir, '{:04d}'.format(video_idx), '{}.jpg'.format(frame_id)))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.zeros((512, 512, 3))

                sam_predictor.set_image(image)

                if frame_id in bboxes:
                    person_bbox = bboxes[frame_id]['person_bbox']
                    object_bbox = bboxes[frame_id]['object_bbox']
                else:
                    person_bbox = None
                    object_bbox = None

                if person_bbox is None and human_in_tracking:
                    person_mask = human_tracker.track(image)
                    person_bbox = extract_bbox_from_mask(person_mask)

                if object_bbox is None and object_in_tracking:
                    object_mask = object_tracker.track(image)
                    object_bbox = extract_bbox_from_mask(object_mask)

                if person_bbox is not None:
                    person_masks, _, _ = sam_predictor.predict(box=np.array(person_bbox[:4]), multimask_output=False)
                    human_tracker.clear_memory()
                    human_tracker.track(image, person_masks[0])
                    human_in_tracking = True
                    person_mask = person_masks[0]
                else:
                    person_mask = None

                if object_bbox is not None:
                    object_masks, _, _ = sam_predictor.predict(box=np.array(object_bbox[:4]), multimask_output=False)
                    object_tracker.clear_memory()
                    object_tracker.track(image, object_masks[0])
                    object_in_tracking = True

                    object_mask = object_masks[0]
                else:
                    object_mask = None

                save_mask(person_mask, object_mask, os.path.join(save_dir, '{}.pkl'.format(frame_id)))
                fine_instance['sequences'].append(
                    {'frame_id': frame_id, 
                     'person_bbox': extract_bbox_from_mask(person_mask) if person_mask is not None else None, 
                     'object_bbox': extract_bbox_from_mask(object_mask) if object_mask is not None else None,})

            fine_tracking_results['hoi_instances'].append(fine_instance)

        save_pickle(os.path.join(fine_tracking_dir, '{:04d}_tracking.pkl'.format(video_idx)), fine_tracking_results)
        print('Video {:04d} Done!'.format(video_idx))


if __name__ == '__main__':
    # envs: sam
    parser = argparse.ArgumentParser(description="Tracking HOI.")
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )
    args = parser.parse_args()

    segment_and_track(args)
