import numpy as np
from scipy.ndimage.measurements import label

from src.window_processing import slide_window_scales, search_windows, add_heat, draw_labeled_bboxes, apply_threshold


class ObjectDetector():
    def __init__(self, pipeline, extract_features, threshold=0.5, moving_avg_length=5):
        self.heat = None
        self.heat_list = []
        self.y_start_stop_new = None
        self.pipeline = pipeline
        self.extract_features = extract_features
        self.current_labels = None
        self.threshold = threshold
        self.moving_avg_length = moving_avg_length

    def process_first(self, image):
        self.heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        self.y_start_stop_new = [image.shape[0] // 2, image.shape[0]]

    def process_image(self, image):
        if self.heat is None or self.y_start_stop_new is None:
            self.process_first(image)

        windows = slide_window_scales(image, sample=None,
                                      x_start_stop=[None, None], y_start_stop=self.y_start_stop_new,
                                      xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        hot_windows = search_windows(image, windows, self.pipeline, self.extract_features)
        current_heat = add_heat(np.zeros_like(self.heat).astype(np.float), hot_windows)
        self.heat_list.append(current_heat)
        self.heat_list = self.heat_list[-self.moving_avg_length:]
        self.average_heatmaps()
        # from collections import deque
        # N= 5  # number of frames to store
        # stored_heat = deque(maxlen = N)
        # ...
        # stored_heat.append(current_frame_heat)

    def average_heatmaps(self):
        self.heat = np.mean(np.array(self.heat_list), axis=0)

    def get_current_objects(self):
        heat_thresh = np.copy(self.heat)
        heat = apply_threshold(heat_thresh, self.threshold)
        heatmap = np.clip(heat, 0, 255)
        self.current_labels = label(heatmap)

    def draw_current_objects(self, image):
        self.get_current_objects()
        return draw_labeled_bboxes(np.copy(image), self.current_labels)
