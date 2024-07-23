import os
import argparse
import numpy as np
import pickle
import math


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def is_valid(args, detections):

    if detections['person'] is None:
        return False
    if detections[args.object_name] is None:
        return False
    if detections['person'][:, -1].max() < 0.3:
        return False
    if detections[args.object_name][:, -1].max() < 0.3 and args.object_name != 'golf' and args.object_name != 'barbell':
        return False
    

    person_bbox = np.array(detections['person'])[:, :4].reshape(-1, 1, 4)
    object_bbox = np.array(detections[args.object_name])[:, :4].reshape(1, -1, 4)

    person_area = (person_bbox[:, :, 2:] - person_bbox[:, :, :2]).prod(-1)
    object_area = (object_bbox[:, :, 2:] - object_bbox[:, :, :2]).prod(-1)
    if person_area.max() < args.min_area:
        return False
    if object_area.max() < args.min_area:
        return False

    union_box_x1y1 = np.minimum(person_bbox[:, :, :2], object_bbox[:, :, :2])
    union_box_x2y2 = np.maximum(person_bbox[:, :, 2:], object_bbox[:, :, 2:])

    inter_box_x1y1 = np.maximum(person_bbox[:, :, :2], object_bbox[:, :, :2])
    inter_box_x2y2 = np.minimum(person_bbox[:, :, 2:], object_bbox[:, :, 2:])

    union_area = (union_box_x2y2 - union_box_x1y1).prod(-1)
    inter_area = (inter_box_x2y2 - inter_box_x1y1).prod(-1).clip(min=0, max=1e8)
    iou = (person_area + object_area - inter_area) / union_area

    if iou.max() < args.min_iou:
        return False

    return True


def split_interval(interval, fps, max_length):
    frame_num = len(interval) * fps
    num_splits = math.ceil(frame_num // max_length / 5)

    split_intervals = []
    for i in range(0, num_splits, len(interval) // num_splits):
        split_intervals.append(interval[i : i + max_length // fps])

    return split_intervals


def filter_frames(args):
    bigdetection_dir = os.path.join(args.root_dir, 'bigdetection_temp')
    for video_idx in range(args.begin_idx, args.end_idx):
        bigdetection_results = load_pickle(os.path.join(bigdetection_dir, '{:04d}_{}.pkl'.format(video_idx, args.interval)))

        all_frames = sorted(list(bigdetection_results.keys()))
        valid_indices = []
        for idx, frame in enumerate(all_frames):
            if is_valid(args, bigdetection_results[frame]):
                if idx > 0 and idx - 1 not in valid_indices:
                        valid_indices.append(idx - 1)

                if idx not in valid_indices:
                    valid_indices.append(idx)

                if idx < len(all_frames) - 1 and idx + 1 not in valid_indices:
                        valid_indices.append(idx + 1)
        valid_indices = sorted(valid_indices)
        valid_intervals = []
        i = 0

        while i < len(valid_indices):
            current_index = valid_indices[i]
            interval = [all_frames[current_index], ]
            while (i + 1) < len(valid_indices) and (current_index + 1) == valid_indices[i + 1]:
                interval.append(all_frames[valid_indices[i + 1]])
                i += 1
                current_index = valid_indices[i]
            valid_intervals.append(interval)
            i += 1

        final_intervals = []
        for interval in valid_intervals:
            if len(interval) * args.interval < args.min_length:
                continue

            if len(interval) * args.interval > args.max_length:
                final_intervals.extend(split_interval(interval, args.interval, args.max_length))
            else:
                final_intervals.append(interval)

        save_pickle(os.path.join(bigdetection_dir, '{:04d}_{}_intervals.pkl'.format(video_idx, args.interval)), final_intervals)
        print('{} Done.'.format('{:04d}_{}_intervals.pkl'.format(video_idx, args.interval)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter Frames.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int, help='The video index')
    parser.add_argument('--end_idx', type=int)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--object_name', type=str,)
    parser.add_argument('--min_length', type=int, default=450, help='at leat have 150 frames per sequence.')
    parser.add_argument('--max_length', type=int, default=360000, help='at most have 3600 frames per sequence.')
    parser.add_argument('--min_area', type=int, default=1000)
    parser.add_argument('--min_iou', type=float, default=0.7)
    args = parser.parse_args()

    filter_frames(args)
