import os
import argparse
import pickle
import numpy as np
from tracker.sort import Sort, linear_assignment


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def iou_batch(bbox1, bbox2):
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    xx1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    union_wh = w * h

    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter_wh = w * h

    o = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) + 
        (bbox2[..., 2] - bbox1[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])) / union_wh

    return o


def pair_hoi(person_dets, object_dets, previous_hoi_dets=None, iou_threshold=0.5):
    if len(person_dets) == 0:
        return np.empty((0, 10))

    person_dets = np.concatenate((person_dets, np.zeros((len(person_dets), 5))), axis=1)
    if len(object_dets) == 0:
        return person_dets

    iou_matrix = iou_batch(person_dets, object_dets)
    matched_indices = linear_assignment(-iou_matrix)
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] > iou_threshold:
            person_id = person_dets[m[0], 4]

            if previous_hoi_dets is not None:
                pervious_object_dets = previous_hoi_dets[previous_hoi_dets[:, 4] == person_id]
                if pervious_object_dets.shape[0] != 0 and pervious_object_dets[0, -1] != 0:
                    object_iou = iou_batch(object_dets[m[1]].reshape(1, -1), pervious_object_dets.reshape(1, -1)[:, 5:])
                    if object_iou[0, 0] < 0.3:
                        continue
            person_dets[m[0], 5:] = object_dets[m[1], :]

    return person_dets


def track_hoi(args):
    root_dir = args.root_dir
    object_name = root_dir.split('/')[-1]
    detection_dir = os.path.join(root_dir, 'bigdetection_temp')

    for video_idx in range(args.begin_idx, args.end_idx):
        if os.path.exists(os.path.join(detection_dir, '{:04d}_track.pkl'.format(video_idx))):
            continue
        try:
            detection_results = load_pickle(os.path.join(detection_dir, '{:04d}_all.pkl'.format(video_idx)))
        except:
            detection_results = []

        tracking_results = {'frames': {}, 'hoi_instances': []}
        final_hoi_id_counter = 0
        for sub_sequence in detection_results:

            tracker = Sort(max_age=10)

            hoi_detections = []
            previous_hoi_dets = None
            for frame_id, detection in sub_sequence.items():

                if detection['person'] is None:
                    person_dets = tracker.update()
                else:
                    person_dets = detection['person']
                    person_dets = person_dets[person_dets[:, -1] > 0.5]
                    if len(person_dets) == 0:
                        person_dets = tracker.update()
                    else:
                        person_dets = tracker.update(person_dets)

                if detection[object_name] is not None:
                    object_dets = detection[object_name]
                    object_dets = object_dets[object_dets[:, -1] > 0.5]
                else:
                    object_dets = np.empty((0, 5))

                # person_dets: [n, 5] (x1, y1, x2, y2, id)
                # object_dets: [m, 5] (x1, y1, x2, y2, score)
                # hoi_dets: [n, 10] (x1, y1, x2, y2, id, x1, y1, x2, y2, score)
                hoi_dets = pair_hoi(person_dets, object_dets, previous_hoi_dets)
                previous_hoi_dets = hoi_dets

                hoi_detections.append(hoi_dets)

            hoi_counts = np.zeros(tracker.tracklet_count)
            person_counts = np.zeros(tracker.tracklet_count)
            for hoi_det in hoi_detections:
                for box in hoi_det:
                    hoi_id = int(box[4])
                    person_counts[hoi_id] += 1
                    if box[-1] != 0:
                        hoi_counts[hoi_id] += 1
            valid_hoi_ids = []
            for hoi_id in range(tracker.tracklet_count):
                if person_counts[hoi_id] > 60 and hoi_counts[hoi_id] / person_counts[hoi_id] > 0.7:
                    valid_hoi_ids.append(hoi_id)

            valid_hoi_instances = {hoi_id: {} for hoi_id in valid_hoi_ids}
            for frame_id, detections in zip(sub_sequence.keys(), hoi_detections):
                for dets in detections:
                    hoi_id = dets[4]
                    if hoi_id in valid_hoi_ids:
                        valid_hoi_instances[hoi_id][frame_id] = {
                            'person_bbox': dets[:5], 
                            'object_bbox': dets[5:] if dets[-1] != 0 else None
                        }

            for valid_hoi_id in valid_hoi_instances:
                tracking_results['hoi_instances'].append({'hoi_id': '{:03d}'.format(final_hoi_id_counter), 'boxes': valid_hoi_instances[valid_hoi_id]})
                final_hoi_id_counter += 1

        all_hoi_instances = tracking_results['hoi_instances']
        for hoi_instances in all_hoi_instances:
            hoi_id = hoi_instances['hoi_id']
            for frame_id, boxes in hoi_instances['boxes'].items():
                if frame_id not in tracking_results['frames']:
                    tracking_results['frames'][frame_id] = []
                tracking_results['frames'][frame_id].append({
                    'hoi_id': hoi_id, 'boxes': boxes
                })

        save_pickle(os.path.join(detection_dir, '{:04d}_track.pkl'.format(video_idx)), tracking_results)
        print('Video {:04d} done !'.format(video_idx))



if __name__ == '__main__':
    # envs bigdet
    parser = argparse.ArgumentParser(description="Tracking HOI.")
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )
    args = parser.parse_args()

    track_hoi(args)
