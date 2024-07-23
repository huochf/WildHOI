import os
import cv2
import json
import pickle
import numpy as np


def main():
    object_name = 'barbell'
    object_kps_dir = './obj_keypoints/{}'.format(object_name)
    object_pose_dir = './object_pose/{}/'.format(object_name)
    object_render_dir = './object_render/{}'.format(object_name)
    root_dir = '/storage/data/huochf/HOIYouTube-test/{}'.format(object_name)

    outputs_dir = './object_render_comparison/{}'.format(object_name)
    os.makedirs(outputs_dir, exist_ok=True)

    for file in os.listdir(object_pose_dir):
        video_id, frame_id, instance_id = file.split('.')[0].split('_')
        rendered_image = cv2.imread(os.path.join(object_render_dir, file.replace('npz', 'jpg')))
        original_image = cv2.imread(os.path.join(root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id)))

        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))
        object_bbox = None
        for hoi_instance in tracking_results['hoi_instances']:
            if hoi_instance['hoi_id'] != instance_id:
                continue
            for item in hoi_instance['sequences']:
                if frame_id == item['frame_id']:
                    object_bbox = item['object_bbox']
        assert object_bbox is not None

        rendered_image = crop_images(rendered_image, object_bbox)
        original_image = crop_images(original_image, object_bbox)

        # try:
        # kps = load_keypoints_v2(os.path.join(object_kps_dir, '{}_{}_{}.json'.format(video_id, frame_id, instance_id)))
        # except:
        #     try:
        #         kps = load_keypoints_v2(os.path.join(object_kps_dir, '{}_{}.json'.format(video_id, frame_id)))
        #     except:
        #         print('File {} not exists.'.format(os.path.join(object_kps_dir, '{}_{}_{}.json'.format(video_id, frame_id, instance_id))))
        #         continue
        kps = load_keypoints(os.path.join(object_kps_dir, '{}_{}_{}.txt'.format(video_id, frame_id, instance_id)))
        # try:
        #     kps = load_keypoints_v2(os.path.join(object_kps_dir, '{}_{}_{}.json'.format(video_id, frame_id, instance_id)))
        # except:
        #     kps = load_keypoints_v2(os.path.join(object_kps_dir, '{}_{}.json'.format(video_id, frame_id)))

        original_image = plot_kps(original_image, kps)

        image_final = np.concatenate([rendered_image, original_image], axis=1)
        cv2.imwrite(os.path.join(outputs_dir, file.replace('npz', 'jpg')), image_final.astype(np.uint8))
        print('{} DONE!'.format(file))


KPS_COLORS = [
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


def plot_kps(image, kps):
    line_thickness = 2
    thickness = 4
    lineType = 8
    h, w, c = image.shape

    for i, point in enumerate(kps):
        x, y, v = point
        if v == 0:
            continue
        x, y = int(x * w), int(y * h)
        cv2.circle(image, (x, y), thickness, KPS_COLORS[i % len(KPS_COLORS)], thickness=-1, lineType=lineType)
        cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return image



def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_keypoints(file):
    with open(file, 'r') as f:
        all_lines = f.readlines()
    if int(all_lines[0]) == 0:
        return np.empty((0, 3))
    keypoints = []
    for i in range(len(all_lines) - 1):
        line = all_lines[i + 1]
        x, y = line.split(' ')
        x, y = float(x), float(y)
        if x > 1 or y > 1 or x < 0 or y < 0:
            keypoints.append([0, 0, 0])
        else:
            keypoints.append([x, y, 1])

    return np.array(keypoints)


def load_keypoints_v2(file):

    with open(file, 'r') as f:
        data = json.load(f)
    data = data
    keypoints = []
    res = data['info']['height']
    for item in data['dataList']:
        x, y = item['coordinates']
        if x > res or y > res:
            keypoints.append([0, 0, 0])
        else:
            keypoints.append([x, y, 1])
    return np.array(keypoints) / res


def crop_images(image, object_bbox):
    h, w, _ = image.shape
    x1, y1, x2, y2 = object_bbox[:4]

    cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
    s = int(max((y2 - y1), (x2 - x1)) * 1.0)
    # s = int(max((y2 - y1), (x2 - x1)) * 1.5) # baseball

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

    return crop_image


if __name__ == '__main__':
    main()
