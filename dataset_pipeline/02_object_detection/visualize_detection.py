import os
import argparse
import pickle
from tqdm import tqdm
import cv2
import numpy as np


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_box(image, box, name):
    thickness = 2
    lineType = 4
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 255, 0], thickness, lineType)

    t_size = cv2.getTextSize(name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    textlbottom = box[0:2] + np.array(list(t_size))
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(textlbottom[0]), int(textlbottom[1])),  [0, 255, 0], -1)
    cv2.putText(image, name , (int(box[0]), int(box[1] + (t_size[1]/2 + 4))), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

    return image


def visualize(args):
    object_name = args.root_dir.split('/')[-1]
    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_id))
    detection_results = os.path.join(args.root_dir, 'bigdetection_temp', '{:04d}_{}.pkl'.format(args.video_id, args.interval))
    detection_results = load_pickle(detection_results)

    intervals = os.path.join(args.root_dir, 'bigdetection_temp', '{:04d}_{}_intervals.pkl'.format(args.video_id, args.interval))
    intervals = load_pickle(intervals)
    print(intervals)

    images = sorted(os.listdir(image_dir))
    image = cv2.imread(os.path.join(image_dir, images[0]))
    h, w, _ = image.shape
    video = cv2.VideoWriter('./big_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for frame_id in tqdm(sorted(list(detection_results.keys()))):
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))

        object_dict = detection_results[frame_id]
        if object_dict['person'] is not None:
            boxes = object_dict['person']
            for box in boxes:
                if box[-1] > 0.3:
                    image = draw_box(image, box[0:4], 'person')
        if object_dict[object_name] is not None:
            boxes = object_dict[object_name]
            for box in boxes:
                if box[-1] > 0.3:
                    image = draw_box(image, box[0:4], object_name)

        video.write(image)
    video.release()

    for i in range(len(intervals)):
        video = cv2.VideoWriter('./big_detection_intervals_{}.mp4'.format(i), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        print('interval {}'.format(i))
        for frame_id in tqdm(intervals[i]):
            image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))

            object_dict = detection_results[frame_id]
            if object_dict['person'] is not None:
                boxes = object_dict['person']
                for box in boxes:
                    if box[-1] > 0.5:
                        image = draw_box(image, box[0:4], 'person')
            if object_dict[object_name] is not None:
                boxes = object_dict[object_name]
                for box in boxes:
                    if box[-1] > 0.5:
                        image = draw_box(image, box[0:4], object_name)

            video.write(image)
        video.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDetection Visualize.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--video_id', type=int)
    parser.add_argument('--interval', type=int)

    args = parser.parse_args()
    visualize(args)
