import atexit
import bisect
import os
import json
import cv2

import torch
import tqdm
import numpy as np
import multiprocessing as mp

from collections import deque

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from filters.kalman import RobustKalmanFilter
from filters.MA import MovingAvgFilter

FONT = {"font.weight": "bold", "font.size": 8}


class VisualizationDemo(object):
    def __init__(self, cfg, parallel, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel
    
        if self.parallel==1:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                instances = self.video_visualizer.draw_instance_bbox(predictions)
                
            return instances

        frame_gen = self._frame_from_video(video)

        if self.parallel==1:
            buffer_size = self.predictor.default_buffer_size
            frame_data = deque()
            
            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)
                
                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)
            
            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:                
                yield process_predictions(frame, self.predictor(frame))




class PoseRefinement(object):
    """
    This class implements keypoints with moving average filter algorithm.
    
    Attributes:
        array (np.array): float matrix of indices
    """
    def __init__(self, width, height, num_frames, basename, window_size):
        self.keypoints_db = {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
        }

        self.width = width
        self.height = height
        self.bbox_min_X = self.width
        self.bbox_min_Y = self.height
        self.bbox_max_X = 0
        self.bbox_max_Y = 0
        self.num_frames = num_frames
        self.basename = str(os.path.splitext(basename)[0])

        self.num_keypoints = 17
        self.num_indices = 0

        self.abs_timeseries_keypoints = {}
        self.maf_timeseries_keypoints = {}

        self.subset_size = window_size

        self.zero_box = np.zeros(shape=[1, 4])
        self.zero_keypoints = np.zeros(shape=[1, self.num_keypoints, 3])

    def noise_reduction_with_MAF(self, json_output_fname):
        """
        Implement noise reduction through Moving Average Filter.
        """
        for i, (key, abs_timeseries_keypoints) in enumerate(self.abs_timeseries_keypoints.items()):
            for num_keypoints in range(self.num_keypoints):
                X_MAF = MovingAvgFilter(
                    abs_timeseries_keypoints[:, num_keypoints, 0].tolist(),
                    abs_timeseries_keypoints[:, num_keypoints, 2].tolist(),
                    subset_size=self.subset_size,
                    avoid_fp_drift=True,
                )
                Y_MAF = MovingAvgFilter(
                    abs_timeseries_keypoints[:, num_keypoints, 1].tolist(),
                    abs_timeseries_keypoints[:, num_keypoints, 2].tolist(),
                    subset_size=self.subset_size,
                    avoid_fp_drift=True,
                )
                
                self.abs_timeseries_keypoints[key][:, num_keypoints, 0] = np.array(list(X_MAF.predict()))
                self.abs_timeseries_keypoints[key][:, num_keypoints, 1] = np.array(list(Y_MAF.predict()))
        result = {"data": []}
        for frame in range(int(self.num_frames)):
            result['data'].append({'frame_index': frame+1, 'skeleton': []})
        
            for i, (key, abs_timeseries_keypoints) in enumerate(self.abs_timeseries_keypoints.items()):
                combined = list(np.r_[[np.round(np.array(abs_timeseries_keypoints[frame,:,0]), 2)],
                                    [np.round(np.array(abs_timeseries_keypoints[frame,:,1]), 2)]].flatten('F'))
                score_list = list(abs_timeseries_keypoints[frame,:,2].round(4))
                result['data'][frame]['skeleton'].append({'pose': combined, 'score': score_list})
        
        with open(os.path.join(json_output_fname + '.json'), 'w') as json_file:
            json.dump(result, json_file)
        
    def extract_keypoints(self, current_frame, box, keypoints, indices):
        """
        Extract timeseries keypoints

        Args:
            boxes(Boxes): :class: `Boxes` object. 1 x 4 shape
            keypoints   : tensor                  1 x 17 x 3
        """

        for i, index in enumerate(indices):
            if not index in self.abs_timeseries_keypoints.keys():
                self.num_indices += 1
                self.abs_timeseries_keypoints[index] = np.around(
                    np.zeros(shape=[self.num_frames, self.num_keypoints, 3]), 4
                )
                
                self.maf_timeseries_keypoints[index] = np.around(
                    np.zeros(shape=[self.num_frames, self.num_keypoints, 3]), 4
                )

            for j, keypoint in enumerate(keypoints[i]):
                pos_X, pos_Y, score = keypoint
                self.abs_timeseries_keypoints[index][current_frame, j, 0] = pos_X
                self.abs_timeseries_keypoints[index][current_frame, j, 1] = pos_Y
                self.abs_timeseries_keypoints[index][current_frame, j, 2] = score


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        print("GPU!", num_workers)
        mp.set_start_method("spawn")

        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
