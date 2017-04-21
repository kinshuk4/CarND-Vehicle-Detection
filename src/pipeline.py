import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm

from moviepy.editor import VideoFileClip
import time

import src.heat_mapper as hmp
import src.feature_extractor as fext

TIME_WINDOW = 20
hot_windows_history = collections.deque(maxlen=TIME_WINDOW)


def combine_output(frame, heatmap, windows):
    screen = np.zeros(frame.shape, dtype=np.uint8)
    screen[0:frame.shape[0], 0:frame.shape[1]] = frame
    screen[frame.shape[0] - 200:frame.shape[0], 0:300] = cv2.resize(heatmap, (300, 200), interpolation=cv2.INTER_AREA)
    for index, window in enumerate(windows):
        screen[30:130, 120 * index:120 * index + 100] = cv2.resize(
            frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], (100, 100), interpolation=cv2.INTER_AREA)
    return screen


def pipeline(frame, svc=None, scaler=None, keep_state=True, verbose=False):
    hot_windows = []

    if verbose is False:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_blurred = hmp.gaussian_blur(frame, 3)

    for size_subsample in np.arange(1, 4, .3):
        hot_windows += fext.find_cars(frame_blurred, 400, 600, size_subsample, svc, scaler)

    if keep_state:
        if hot_windows:
            hot_windows_history.append(hot_windows)
            hot_windows = np.concatenate(hot_windows_history)

    thresh = TIME_WINDOW if keep_state else 1
    heatmap, heatmap_thresh = hmp.compute_heatmap_from_detections(frame, hot_windows, threshold=thresh, verbose=False)

    img_thresh = np.zeros(heatmap_thresh.shape, dtype=np.uint8)
    img_blur = hmp.gaussian_blur(heatmap_thresh, 21)
    img_thresh[img_blur > 50] = 255

    img_hot_windows = hmp.draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=2)
    img_heatmap = cv2.applyColorMap(hmp.normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)
    img_labeling = hmp.normalize_image(img_thresh)
    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for i, contour in enumerate(contours):
        rectangle = cv2.boundingRect(contour)
        x, y, w, h = rectangle
        boxes.append([(x, y), (x + w, y + h)])

    img_detection = hmp.draw_boxes(frame, boxes, thick=2, color=(255, 0, 0))

    if verbose:
        plt.figure(figsize=(16, 10))
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(img_hot_windows, cv2.COLOR_BGR2RGB))
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB))
        plt.subplot(223)
        plt.imshow(img_labeling, cmap='gray')
        plt.subplot(224)
        plt.imshow(cv2.cvtColor(img_detection, cv2.COLOR_BGR2RGB))
    else:
        return combine_output(cv2.cvtColor(img_detection, cv2.COLOR_BGR2RGB),
                              cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB), boxes)


def video_pipeline(input_file="project_video.mp4", output_file='out_project_video.mp4', svc=None, scaler=None,
                   keep_state=True, verbose=False):
    clip1 = VideoFileClip(input_file)

    def image_pipeline(frame, svc=svc, scaler=scaler, keep_state=keep_state, verbose=verbose):
        return pipeline(frame, svc=svc, scaler=scaler, keep_state=keep_state, verbose=verbose)

    out_clip = clip1.fl_image(image_pipeline)
    return out_clip.write_videofile(output_file, audio=False, verbose=True, progress_bar=False)


def main():
    pass


if __name__ == '__main__':
    main()
