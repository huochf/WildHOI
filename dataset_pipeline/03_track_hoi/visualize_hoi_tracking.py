import os
import argparse
import pickle
from tqdm import tqdm
import cv2
import numpy as np
from utils.mask_painter import mask_hoi_painter

points_colors = [
    [0.,    255.,  255.],
    [0.,   255.,    170.],
    [0., 170., 255.,],
    [85., 170., 255.],
    [0.,   255.,   85.], # 4
    [0., 85., 255.],
    [170., 85., 255.],
    [0.,   255.,   0.], # 7
    [0., 0., 255.], 
    [255., 0., 255.],
    [0.,    255.,  0.], # 10
    [0., 0., 255.],
    [255., 85., 170.],
    [170., 0, 255.],
    [255., 0., 170.],
    [255., 170., 85.],
    [85., 0., 255.],
    [255., 0., 85],
    [32., 0., 255.],
    [255., 0, 32],
    [0., 0., 255.],
    [255., 0., 0.],
]


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_box(image, box, name, color):
    thickness = 2
    lineType = 4
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness, lineType)

    t_size = cv2.getTextSize(name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    textlbottom = box[0:2] + np.array(list(t_size))
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(textlbottom[0]), int(textlbottom[1])),  [0, 0, 0], -1)
    cv2.putText(image, name , (int(box[0]), int(box[1] + (t_size[1]/2 + 4))), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

    return image


def visualize(args):
    tracking_results = os.path.join(args.root_dir, 'hoi_tracking', '{:04d}_tracking.pkl'.format(args.video_idx))
    tracking_results = load_pickle(tracking_results)

    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx))
    mask_dir = os.path.join(args.root_dir, 'hoi_mask', '{:04d}'.format(args.video_idx))

    # print(tracking_results)
    for hoi_instance in tracking_results['hoi_instances']:
        hoi_id = hoi_instance['hoi_id']
        frame_id = hoi_instance['sequences'][0]['frame_id']
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        h, w, _ = image.shape
        # h, w = 1080, 1920
        # h, w = 720, 1280
        os.makedirs('./__debug__')
        video = cv2.VideoWriter('./__debug__/tracking_hoi_{}.mp4'.format(hoi_id), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        for item in tqdm(hoi_instance['sequences']):
            frame_id = item['frame_id']
            person_bbox = item['person_bbox']
            object_bbox = item['object_bbox']

            image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))

            masks = load_pickle(os.path.join(mask_dir, '{}'.format(hoi_id), '{}.pkl'.format(frame_id)))
            person_mask = np.zeros((h, w))
            if masks['human']['mask'] is not None:
                mask_h, mask_w = masks['human']['mask_shape']
                x1, y1, x2, y2 = masks['human']['mask_box']
                person_mask[y1:y2+1, x1:x2+1] = np.unpackbits(masks['human']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

            object_mask = np.zeros((h, w))
            if masks['object']['mask'] is not None:
                mask_h, mask_w = masks['object']['mask_shape']
                x1, y1, x2, y2 = masks['object']['mask_box']
                object_mask[y1:y2+1, x1:x2+1] = np.unpackbits(masks['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

            image = mask_hoi_painter(image, person_mask.astype(np.uint8), object_mask.astype(np.uint8), background_alpha=0.6)

            if person_bbox is not None:
                image = draw_box(image, person_bbox[:4], str(hoi_id), points_colors[int(hoi_id) % len(points_colors)])
                
            if object_bbox is not None:
                image = draw_box(image, object_bbox[:4], str(hoi_id), points_colors[int(hoi_id) % len(points_colors)])

            video.write(image)
        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracking HOI.")
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--video_idx', type=int, )
    args = parser.parse_args()

    visualize(args)
