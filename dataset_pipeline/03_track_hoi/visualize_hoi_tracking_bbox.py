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
    object_name = args.root_dir.split('/')[-1]

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        tracking_results = os.path.join(args.root_dir, 'bigdetection_temp', '{:04d}_track.pkl'.format(video_idx))
        tracking_results = load_pickle(tracking_results)

        image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(video_idx))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            bboxes = hoi_instance['boxes']

            all_frames = sorted(bboxes.keys())
            frame_begin_idx = int(all_frames[0])
            frame_end_idx = int(all_frames[-1])

            image = cv2.imread(os.path.join(image_dir, '{:06d}.jpg'.format(frame_begin_idx)))
            if image is None:
                continue

            h, w, _ = image.shape
            os.makedirs(os.path.join(args.save_dir, '{}_sequences'.format(object_name)))
            video = cv2.VideoWriter(os.path.join(args.save_dir, '{}_sequences'.format(object_name), 'tracking_hoi_{}_{}.mp4'.format(video_id, hoi_id)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

            for frame_idx in tqdm(range(frame_begin_idx, frame_end_idx)):
                frame_id = '{:06d}'.format(frame_idx)
                image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
                if image is None:
                    continue

                if frame_id in bboxes:
                    person_bbox = bboxes[frame_id]['person_bbox']
                    object_bbox = bboxes[frame_id]['object_bbox']

                    if person_bbox is not None:
                        image = draw_box(image, person_bbox[:4], str(hoi_id), points_colors[int(hoi_id) % len(points_colors)])
                        
                    if object_bbox is not None:
                        image = draw_box(image, object_bbox[:4], str(hoi_id), points_colors[int(hoi_id) % len(points_colors)])

                video.write(image)
            video.release()

        print('Video {} done.'.format(video_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracking HOI.")
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )
    parser.add_argument('--save_dir', type=str, )
    args = parser.parse_args()

    visualize(args)
