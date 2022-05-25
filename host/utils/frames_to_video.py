import cv2
import glob

import numpy as np


def frames_to_video(frames, fps, video_path):
    """
    Converts a list of frames to a video.

    Args:
        frames: list of frames to convert to video
        fps: frames per second
        video_path: path to save video to
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bin_out = cv2.VideoWriter(f'{video_path}_binary.mp4', fourcc, fps, (128, 128))
    color_out = cv2.VideoWriter(f'{video_path}_color.mp4', fourcc, fps, (128, 128))
    for frame_p in sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1])):
        frame_normal = cv2.imread(frame_p, cv2.IMREAD_GRAYSCALE)
        # binarize frame
        _, frame = cv2.threshold(frame_normal, 52, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        k_open_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        k_open_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, k_open_rect)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, k_open_ellipse)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, k_close_ellipse)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, k_close_rect)
        color_out.write(frame_normal)
        bin_out.write(frame)
    color_out.release()
    bin_out.release()


if __name__ == '__main__':
    frames = glob.glob('../../frames/*.png')
    frames_to_video(frames, 20, 'out')
    print('Done')
    exit(0)
