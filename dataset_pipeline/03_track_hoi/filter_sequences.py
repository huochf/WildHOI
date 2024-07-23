import os
import json
import pickle
from tqdm import tqdm


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def select_frames():

    with open('./barbell_sequences.json', 'r') as f:
        sequences = json.load(f)

    need_to_split_sequences = sequences['need_to_split']
    sort_tracking_dir = '/storage/data/huochf/HOIYouTube-test/barbell/bigdetection_temp'
    image_dir = '/storage/data/huochf/HOIYouTube-test/barbell/images_temp'
    out_dir = '/storage/data/huochf/working_dir/barbell_sequences'

    print(need_to_split_sequences)

    for video_idx in range(0, 31):
        sort_tracking_results = os.path.join(sort_tracking_dir, '{:04d}_track.pkl'.format(video_idx))
        sort_tracking_results = load_pickle(sort_tracking_results)
        video_id = '{:04d}'.format(video_idx)
        for hoi_instance in sort_tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']

            if [video_id, hoi_id] in need_to_split_sequences:
                bboxes = hoi_instance['boxes']

                all_frames = sorted(bboxes.keys())
                frame_begin_idx = int(all_frames[0])
                frame_end_idx = int(all_frames[-1])
                for frame_idx in tqdm(range(frame_begin_idx, frame_end_idx)):
                    frame_id = '{:06d}'.format(frame_idx)
                    source_image = os.path.join(image_dir, '{:04d}'.format(video_idx), '{}.jpg'.format(frame_id))
                    target_image = os.path.join(out_dir, '{:04d}_{}'.format(video_idx, hoi_id))
                    os.makedirs(target_image, exist_ok=True)
                    target_image = os.path.join(target_image, '{}.jpg'.format(frame_id))
                    cmd = 'cp {} {}'.format(source_image, target_image)
                    os.system(cmd)


def update_sort_tracking(args):

    object_name = args.root_dir.split('/')[-1]
    with open('./{}_sequences.json'.format(object_name), 'r') as f:
        sequences = json.load(f)

    # if os.path.exists(os.path.join('./golf_select_sequences.json')):
    #     with open('./golf_select_sequences.json', 'r') as f:
    #         selected_frames = json.load(f)
    # else:
    selected_frames = []

    need_to_split_sequences = sequences['need_to_split']
    good_sequences = sequences['good']
    sort_tracking_dir = os.path.join(args.root_dir, 'bigdetection_temp')

    for video_idx in range(0, 31):
        sort_tracking_results = os.path.join(sort_tracking_dir, '{:04d}_track.pkl'.format(video_idx))
        sort_tracking_results = load_pickle(sort_tracking_results)
        video_id = '{:04d}'.format(video_idx)

        filtered_sort_tracking_results = []
        for hoi_instance in sort_tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']

            if [video_id, hoi_id] in good_sequences:
                filtered_sort_tracking_results.append(hoi_instance)
            elif [video_id, hoi_id] in need_to_split_sequences and '{}_{}'.format(video_id, hoi_id) in selected_frames:
                frames = selected_frames['{}_{}'.format(video_id, hoi_id)]
                new_hoi_instances = {'hoi_id': hoi_id, 'boxes': {}}
                bboxes = hoi_instance['boxes']

                for frame_id in bboxes:
                    if frame_id in frames:
                        new_hoi_instances['boxes'][frame_id] = bboxes[frame_id]
                filtered_sort_tracking_results.append(new_hoi_instances)
            else:
                print('Delect sequence {}_{}.'.format(video_id, hoi_id))

        save_pickle(os.path.join(sort_tracking_dir, '{:04d}_track.pkl'.format(video_idx)), 
            {'hoi_instances': filtered_sort_tracking_results}, )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean images.")
    parser.add_argument('--root_dir', type=str, )

    args = parser.parse_args()
    # select_frames()
    update_sort_tracking(args)
