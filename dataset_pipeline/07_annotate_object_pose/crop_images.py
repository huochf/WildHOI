import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def crop_images():
    object_name = 'surfboard'
    outputs_dir = './images/{}'.format(object_name)
    os.makedirs(outputs_dir, exist_ok=True)
    transform = transforms.ToTensor()

    num_image_per_video = 20
    root_dir = '/storage/data/huochf/HOIYouTube/{}'.format(object_name)

    count = 0

    for video_idx in range(0, 600):
        video_id = '{:04d}'.format(video_idx)

        if not os.path.exists(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id))):
            print('Skip {}.'.format(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id))))
            continue

        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']

            sequences = hoi_instance['sequences']

            item = sequences[np.random.randint(len(sequences))]

            frame_id = item['frame_id']

            object_bbox = item['object_bbox']
            if object_bbox is None:
                continue

            image = cv2.imread(os.path.join('/storage/data/huochf/HOIYouTube', object_name, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
            h, w, _ = image.shape
            x1, y1, x2, y2 = object_bbox[:4]

            cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
            # s = int(max((y2 - y1), (x2 - x1)) * 1.0)
            s = int(max((y2 - y1), (x2 - x1)) * 1.5)

            crop_image = np.zeros((2 * s, 2 * s, 3))

            _x1 = int(cx - s)
            _y1 = int(cy - s)

            if _x1 < 0 and _y1 < 0:
                crop_image[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[0:_y1 + 2 * s, 0:_x1 + 2 * s]
            elif _x1 < 0:
                crop_image[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
            elif _y1 < 0:
                crop_image[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
            else:
                crop_image[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

            crop_image = cv2.resize(crop_image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            crop_image = cv2.rectangle(crop_image, (0, 0), (1024, 1024), (0, 0, 255), 10)

            output_path = os.path.join(outputs_dir, '{}_{}_{}.jpg'.format(video_id, frame_id, hoi_id))
            cv2.imwrite(output_path, crop_image.astype(np.uint8))
            print('write image to path: {}'.format(output_path))
            # if count > 50:
            #     exit(0)


def crop_images2():
    object_name = 'barbell'
    outputs_dir = './images/{}'.format(object_name)
    os.makedirs(outputs_dir, exist_ok=True)
    transform = transforms.ToTensor()

    root_dir = '/storage/data/huochf/HOIYouTube-test/{}'.format(object_name)
    count = 0

    with open('./barbell_reanno_list.json', 'r') as f:
        reanno_list = json.load(f)

    for img_id in tqdm(reanno_list):

        # video_id, frame_id, hoi_id = img_id.split('_')
        video_id, hoi_id, frame_id = img_id.split('_')

        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))

        object_bbox = None
        for hoi_instance in tracking_results['hoi_instances']:
            if hoi_instance['hoi_id'] != hoi_id:
                continue

            for item in hoi_instance['sequences']:
                if frame_id == item['frame_id']:
                    object_bbox = item['object_bbox']

            sequences = hoi_instance['sequences']

            item = sequences[np.random.randint(len(sequences))]
        if object_bbox is None:
            continue
        assert object_bbox is not None

        image = cv2.imread(os.path.join('/storage/data/huochf/HOIYouTube-test', object_name, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
        h, w, _ = image.shape
        x1, y1, x2, y2 = object_bbox[:4]

        cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)
        # s = int(max((y2 - y1), (x2 - x1)) * 1.5)

        crop_image = np.zeros((2 * s, 2 * s, 3))

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        if _x1 < 0 and _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[0:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _x1 < 0:
            crop_image[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
        else:
            crop_image[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

        crop_image = cv2.resize(crop_image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        crop_image = cv2.rectangle(crop_image, (0, 0), (1024, 1024), (0, 0, 255), 10)

        output_path = os.path.join(outputs_dir, '{}_{}_{}.jpg'.format(video_id, frame_id, hoi_id))
        cv2.imwrite(output_path, crop_image.astype(np.uint8))
        print('[{:04d}] write image to path: {}'.format(count, output_path))
        count += 1


if __name__ == '__main__':
    # crop_images()
    crop_images2()
