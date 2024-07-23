import os
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    object_name = 'barbell'
    data_root_dir = '/storage/data/huochf/HOIYouTube-test/{}/'.format(object_name)
    pose_anno_dir = os.path.join(data_root_dir, 'object_annotations', 'pose')
    kps_anno_dir = os.path.join(data_root_dir, 'object_annotations', 'keypoints')

    os.makedirs(pose_anno_dir, exist_ok=True)
    os.makedirs(kps_anno_dir, exist_ok=True)

    data_valid_dir = './object_render_comparison/{}_valid/'.format(object_name)
    data_kps_dir = './obj_keypoints/{}/'.format(object_name)
    data_pose_dir = './object_pose/{}/'.format(object_name)

    valid_count = 0
    for file in os.listdir(data_valid_dir):
        with open(os.path.join(data_valid_dir, file), 'r') as f:
            all_files = f.readlines()

        if int(all_files[0]) == 0:
            video_id, frame_id, instance_id = file.split('.')[0].split('_')
            # if os.path.exists(os.path.join(data_kps_dir, '{}_{}.json'.format(video_id, frame_id))):
            #     kps_source = os.path.join(data_kps_dir, '{}_{}.json'.format(video_id, frame_id, ))
            # else:
            # kps_source = os.path.join(data_kps_dir, '{}_{}_{}.json'.format(video_id, frame_id, instance_id))
            kps_source = os.path.join(data_kps_dir, '{}_{}_{}.txt'.format(video_id, frame_id, instance_id))
            pose_source = os.path.join(data_pose_dir, '{}_{}_{}.npz'.format(video_id, frame_id, instance_id))

            hoi_tracking_results = load_pickle(os.path.join(data_root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))

            for item in hoi_tracking_results['hoi_instances']:
                if item['hoi_id'] == instance_id:
                    instance = item

            kps_target = os.path.join(kps_anno_dir, '{:04d}_{:06d}_{:03d}.json'.format(int(video_id), int(frame_id), int(instance_id)))
            pose_target = os.path.join(pose_anno_dir, '{:04d}_{:06d}_{:03d}.npz'.format(int(video_id), int(frame_id), int(instance_id)))
            cmd = 'mv {} {}'.format(kps_source, kps_target)
            os.system(cmd)
            print(cmd)

            cmd = 'mv {} {}'.format(pose_source, pose_target)
            os.system(cmd)
            print(cmd)

            valid_count += 1

    print('valid_count: ', valid_count)


def move_files():
    object_name = 'bicycle'
    obj_anno_dir = '/storage/data/huochf/HOIYouTube/{}/_object_annotations'.format(object_name)
    pose_file_list = list(os.listdir(os.path.join(obj_anno_dir.replace('_object_annotations', 'object_annotations'), 'pose')))
    print(len(pose_file_list))
    count = 0

    print(len(list(os.listdir(os.path.join(obj_anno_dir, 'pose')))))
    # exit(0)
    os.makedirs('./object_pose/{}'.format(object_name), exist_ok=True)
    for file in os.listdir(os.path.join(obj_anno_dir, 'pose')):
        item_id = file.split('.')[0]
        if file not in pose_file_list:
            # print(file)
            count += 1
            source_file = os.path.join(obj_anno_dir, 'pose', file)
            target_file = os.path.join('./object_pose/{}'.format(object_name), file)
            cmd = 'cp {} {}'.format(source_file, target_file)
            os.system(cmd)
            print(cmd)

            # source_file = os.path.join(obj_anno_dir, 'keypoints', file.replace('npz', 'txt'))
            # if not os.path.exists(source_file):
            #     item_id = '_'.join(file.split('_')[:2])
            #     source_file = os.path.join(obj_anno_dir, 'keypoints', '{}.txt'.format(item_id))
            # target_file = os.path.join('./obj_keypoints_v2/{}'.format(object_name), file)
            # cmd = 'cp {} {}'.format(source_file, target_file)
            # os.system(cmd)
            # print(cmd)
    print(count)




if __name__ == '__main__':
    main()
    # move_files()
