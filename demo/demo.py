import argparse
import os
import sys
import cv2
import time
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.pose_box_correction import keypoint_correction

from predictor import VisualizationDemo, PoseRefinement


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--parallel",
        type=int,
        help="Setting model with multi-gpus",
        default=0,
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))


    cfg = setup_cfg(args)

    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        basename = os.path.basename(args.video_input)
        output_folder_name = os.path.join("../output/", os.path.splitext(basename)[0])
        os.makedirs(output_folder_name)
        if args.output:
            json_output_fname = os.path.join(args.output)[:-4]

        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        demo = VisualizationDemo(cfg=cfg, parallel=args.parallel)

        window_size = 5
        
        # make video with bbox and append instance information
        data_array = []
        idx = 0
        for frame_instances in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            idx += 1
            data_array.append(frame_instances)
        video.release()
        
        num_frames = idx
        pose_refinement = PoseRefinement(
            width, height, num_frames, basename, window_size
        )
        
        # correction keypoints - using adjacent frame information
        for i, frame_instances in tqdm.tqdm(enumerate(data_array), total=len(data_array)):
            boxes = [inst.bbox.tolist() for inst in frame_instances]
            keypoints = [inst.keypoint.tolist() for inst in frame_instances]
            indices = [inst.index for inst in frame_instances]
            if i >= window_size and i < len(data_array) - window_size:
                keypoints = keypoint_correction(
                    data_array[i - window_size : i + window_size + 1], window_size
                )
            pose_refinement.extract_keypoints(i, boxes, keypoints, indices)
        del data_array
        # Normalize based on updated relative coordinate based on bounding boxes
        pose_refinement.noise_reduction_with_MAF(json_output_fname)
