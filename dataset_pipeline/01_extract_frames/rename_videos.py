import os
import argparse


def rename_videos(args):
    all_videos = os.listdir(args.video_dir)
    for idx, video in sorted(enumerate(all_videos)):
        cmd = 'mv {} {}'.format(os.path.join(args.video_dir, video), os.path.join(args.video_dir, '{:04d}.{}'.format(idx, video.split('.')[-1])))
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename Videos.')
    parser.add_argument('--video_dir', type=str, help="The video directory")
    args = parser.parse_args()
    rename_videos(args)
