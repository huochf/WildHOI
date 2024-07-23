import os
import argparse
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def clean(args):
    image_dir = os.path.join(args.root_dir, 'images_temp')

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        coarse_tracking_results = load_pickle(os.path.join(args.root_dir, 'bigdetection_temp', '{:04d}_track.pkl'.format(video_idx)))

        print('Find {} HOI instances.'.format(len(coarse_tracking_results['hoi_instances'])))
        useful_frames = {}

        for hoi_instance in coarse_tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            bboxes = hoi_instance['boxes']

            all_frames = sorted(bboxes.keys())
            if len(all_frames) == 0:
                continue
            frame_begin_idx = int(all_frames[0])
            frame_end_idx = int(all_frames[-1])

            for frame_idx in range(frame_begin_idx, frame_end_idx):
                frame_id = '{:06d}'.format(frame_idx)
                useful_frames[frame_id] = True

        for file in os.listdir(os.path.join(image_dir, video_id, )):
            if file.split('.')[0] not in useful_frames:
                cmd = 'rm {}'.format(os.path.join(image_dir, video_id, file))
                os.system(cmd)
                print(cmd)

        print('Video {} done.'.format(video_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean images.")
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--begin_idx', type=int, )
    parser.add_argument('--end_idx', type=int, )
    args = parser.parse_args()
    clean(args)
