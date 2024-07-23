import os
import argparse
from tqdm import tqdm


def extract_frames(args):
    video_dir = os.path.join(args.root_dir, 'videos')
    image_dir = os.path.join(args.root_dir, 'images_temp')
    os.makedirs(image_dir, exist_ok=True)

    for i in tqdm(range(args.begin_idx, args.end_idx)):
        video_path = os.path.join(video_dir, '{:04d}.mp4'.format(i))
        image_path = os.path.join(image_dir, '{:04d}'.format(i))
        os.makedirs(image_path, exist_ok=True)
        cmd = 'ffmpeg -i {} {}/%06d.jpg'.format(video_path, image_path)
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    # in preprocess envs.
    parser = argparse.ArgumentParser(description='Rename Videos.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()
    extract_frames(args)
