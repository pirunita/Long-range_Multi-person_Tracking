# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import pycocotools.mask as mask_util

from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
)

from .colormap import random_color


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        index (int): index number for the instance.
        path (list[float]): trajectory for the instance.
        extra (bool): if extra=True, the previously detected instance is now unseen
        hide (bool): extrapolation for the instance. For example, if hide=True,
            the instance is unseen so corresponding bbox is extrapolated.
        hide_time (int): time for extrapolation.
        overlap (bool): overlap for the instance.
        keypoint (list[float]): keypoint for instance.
        sit (bool): if sit=True, the instance is in sit status.
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = [
        "label",
        "bbox",
        "index",
        "path",
        "extra",
        "hide",
        "hide_time",
        "overlap",
        "keypoint",
        "sit",
        "mask_rle",
        "color",
        "ttl",
    ]

    def __init__(
        self,
        label,
        bbox,
        index,
        path,
        extra,
        hide,
        hide_time,
        overlap,
        keypoint,
        sit,
        mask_rle,
        color,
        ttl,
    ):
        self.label = label
        self.bbox = bbox
        self.index = index
        self.path = path
        self.extra = extra
        self.hide = hide
        self.hide_time = hide_time
        self.overlap = overlap
        self.keypoint = keypoint
        self.sit = sit
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self.index = 0
        self.area_threshold = 3000
        self.sit_threshold = 75
        self.overlap_threshold = 0.1
        self.invalid_bbox_threshold = 0.05

    def draw_instance_bbox(self, predictions):
        
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(predictions)
        # If there is no newly detected instance, return old instances 
        # which are detected at previous frame
        if num_instances == 0:
            for idx, inst in enumerate(self._old_instances):
                inst.ttl -= 1
                if inst.ttl <= 0:
                    del self._old_instances[idx]
            boxes = [inst.bbox.tolist() for inst in self._old_instances]
            colors = [inst.color for inst in self._old_instances]
            indices = [inst.index for inst in self._old_instances]
            labels = _create_text_labels(
                None, indices, self.metadata.get("thing_classes", None)
            )
            
            return self._old_instances

        boxes = (
            predictions.pred_boxes.tensor.numpy()
            if predictions.has("pred_boxes")
            else None
        )
        classes = (
            predictions.pred_classes.numpy()
            if predictions.has("pred_classes")
            else None
        )
        keypoints = (
            predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        )
        
        # Detect small box which area is small than area threshold
        del_idx = []
        for idx, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < self.area_threshold:
                num_instances -= 1
                del_idx.append(idx)
        del_idx.reverse()
        for _, idx in enumerate(del_idx):
            boxes = np.delete(boxes, idx, 0)
            classes = np.delete(classes, idx, 0)
            keypoints = np.delete(keypoints, idx, 0)
            
        # If all of newly instance is smaller than area_threshold,
        # return old instances which are detected at previous frame
        if num_instances == 0:
            for idx, inst in enumerate(self._old_instances):
                inst.ttl -= 1
                if inst.ttl <= 0:
                    del self._old_instances[idx]
            boxes = [inst.bbox.tolist() for inst in self._old_instances]
            colors = [inst.color for inst in self._old_instances]
            indices = [inst.index for inst in self._old_instances]
            labels = _create_text_labels(
                None, indices, self.metadata.get("thing_classes", None)
            )
            
            return self._old_instances

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        detected = [
            _DetectedInstance(
                classes[i],
                bbox=boxes[i],
                index=None,
                path=[boxes[i]],
                extra=False,
                hide=False,
                hide_time=1,
                overlap=False,
                keypoint=keypoints[i],
                sit=False,
                mask_rle=None,
                color=None,
                ttl=50,
            )
            for i in range(num_instances)
        ]
        colors, indices = self.tracking(detected)
        labels = _create_text_labels(
            classes, indices, self.metadata.get("thing_classes", None)
        )
        boxes = [inst.bbox.tolist() for inst in self._old_instances]

        if self._instance_mode == ColorMode.IMAGE_BW:
            alpha = 0.3
        else:
            alpha = 0.5

        # The function which return true when first point is at higher location than other points.
        # For example, *points=head, wrist, ankle, it will return true in normal case
        def isHigh(*points):
            std = points[0][1]
            for point in points:
                if std > point[1]:
                    return False
            return True

        # Calculate a degree between point1, point2 and point3
        def calDegree(point1, point2, point3):
            a = point1[:2]
            b = point2[:2]
            c = point3[:2]

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) + 1e-6 / (
                (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
            )
            if cosine_angle < -1:
                cosine_angle = -1.0
            if cosine_angle > 1:
                cosine_angle = 1.0

            angle = np.degrees(np.arccos(cosine_angle))

            return angle
        
        for idx, inst in enumerate(self._old_instances):
            # Detect whether each keypoint is located at plausible position or not
            # For example, shoulder should be located higher than writst, knee, foot
            if not isHigh(
                inst.keypoint[5],
                inst.keypoint[11],
                inst.keypoint[12],
                inst.keypoint[13],
                inst.keypoint[14],
                inst.keypoint[15],
                inst.keypoint[16],
            ):
                inst.keypoint[5][2] = 0
            if not isHigh(
                inst.keypoint[6],
                inst.keypoint[11],
                inst.keypoint[12],
                inst.keypoint[13],
                inst.keypoint[14],
                inst.keypoint[15],
                inst.keypoint[16],
            ):
                inst.keypoint[6][2] = 0

            # wrist should be located higher than foot
            if not isHigh(inst.keypoint[11], inst.keypoint[15], inst.keypoint[16]):
                inst.keypoint[11][2] = 0
            if not isHigh(inst.keypoint[12], inst.keypoint[15], inst.keypoint[16]):
                inst.keypoint[12][2] = 0

            # Detect whether instance sit or not
            left_wrist = inst.keypoint[11, :]
            left_knee = inst.keypoint[13, :]
            left_ankle = inst.keypoint[15, :]
            right_wrist = inst.keypoint[12, :]
            right_knee = inst.keypoint[14, :]
            right_ankle = inst.keypoint[16, :]

            if (
                calDegree(left_wrist, left_knee, left_ankle) < self.sit_threshold
                or calDegree(right_wrist, right_knee, right_ankle) < self.sit_threshold
            ):
                inst.sit = True

        return self._old_instances

    def draw_instance_keypoints(self, frame, keypoints):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)

        frame_visualizer.overlay_instances(
            boxes=None,  # boxes are a bit distracting
            masks=None,
            labels=None,
            keypoints=keypoints,
            assigned_colors=None,
            alpha=0.3,
        )

        return frame_visualizer.output

    def tracking(self, instances):
        """
        Naive tracking heuristics
        to assign same color to the same instance,
        to detect hide,
        to extrapolation.

        Returns:
            list[tuple[float]]: list of colors.
            list[tuple[float]]: list of indices.
        """

        # Compute iou with either boxes or masks:
        def computeIoU(bboxes_1, bboxes_2):
            ious = np.zeros((len(bboxes_1), len(bboxes_2)), dtype="float32")

            def bb_intersection_over_union(boxA, boxB):
                # determine the (x, y)-coordinates of the intersection rectangle
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                # return the intersection over union value
                return round(iou, 2)

            for idx1, box1 in enumerate(bboxes_1):
                for idx2, box2 in enumerate(bboxes_2):
                    ious[idx1][idx2] = bb_intersection_over_union(box1, box2)

            return ious

        # Compute iou with either boxes or masks:
        boxes_old = [x.bbox for x in self._old_instances]
        boxes_new = [x.bbox for x in instances]
        ious = computeIoU(boxes_old, boxes_new)
        threshold = 0.2
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    instances[newidx].index = inst.index
                    instances[newidx].path = np.append(
                        inst.path, instances[newidx].path, axis=0
                    )
                    if inst.ttl > 50:
                        instances[newidx].ttl = inst.ttl
                    # If some instances stay at same location while long time,
                    # they will also stay after this frame
                    # So increasing ttl of those instances
                    if len(inst.path) > 40:
                        if (
                            (
                                (instances[newidx].bbox[0] - inst.bbox[0]) ** 2
                                + (instances[newidx].bbox[1] - inst.bbox[1]) ** 2
                            )
                            ** 0.5
                        ) < (inst.bbox[2] - inst.bbox[0]) / 10:
                            instances[newidx].ttl = inst.ttl + 1
                            max_ttl = 80
                            if instances[newidx].ttl > max_ttl:
                                instances[newidx].ttl = max_ttl
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                if inst.color is not None:
                    inst.extra = True
                    extra_instances.append(inst)

        # Try to find instances which is overlapped with other instances
        for newidx, _ in enumerate(instances):
            overlap_num = 0
            for oldidx, _ in enumerate(self._old_instances):
                if not self._old_instances[oldidx].hide:
                    if ious[oldidx][newidx] > self.overlap_threshold:
                        overlap_num += 1
                        if overlap_num > 1:
                            instances[newidx].overlap = True

        def isInside(_old_instances, _inst):
            box = _inst.bbox
            for _, _old_instance in enumerate(_old_instances):
                old_box = _old_instance.bbox
                if (
                    old_box[0] < box[0]
                    and old_box[1] < box[1]
                    and old_box[2] > box[2]
                    and old_box[3] > box[3]
                ):
                    if _old_instance.color is not None:
                        return True
            return False

        # Assign random color to newly-detected instances:
        del_idx = []
        for idx, inst in enumerate(instances):
            if inst.color is None:
                # If some boxes are newly detected but overlapped with existed boxes, remove them
                if len(ious) > 0 and np.max(ious[:, idx]) > self.invalid_bbox_threshold:
                    if self._old_instances[np.argmax(ious[:, idx])].color is not None:
                        del_idx.append(idx)
                        continue
                # If some boxes are newly detected within existed boxes, remove them
                if isInside(self._old_instances, inst):
                    del_idx.append(idx)
                    continue
                inst.color = random_color(rgb=True, maximum=1)
                inst.index = self.index
                self.index += 1
        del_idx.reverse()
        for _, idx in enumerate(del_idx):
            del instances[idx]
        self._old_instances = instances[:] + extra_instances

        def extrapolation_bbox(_instance):
            t = len(_instance.path)
            time = np.arange(1, t + 1)
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            for _path in _instance.path:
                x1.append(_path[0])
                y1.append(_path[1])
                x2.append(_path[2])
                y2.append(_path[3])
            fp1 = np.polyfit(time, x1, 2)
            f1 = np.poly1d(fp1)
            fp2 = np.polyfit(time, y1, 2)
            f2 = np.poly1d(fp2)

            if abs(fp1[1] + fp2[1]) < 0.9:
                result_bbox = np.array(
                    [
                        _instance.path[-1][0],
                        _instance.path[-1][1],
                        _instance.path[-1][2],
                        _instance.path[-1][3],
                    ]
                )
                for i in range(17):
                    _instance.keypoint[i, 2] = -1
                return result_bbox, _instance.keypoint

            x = f1(t + _instance.hide_time)
            y = f2(t + _instance.hide_time)
            width = _instance.bbox[2] - _instance.bbox[0]
            height = _instance.bbox[3] - _instance.bbox[1]

            result_bbox = np.array([x, y, x + width, y + height])

            for i in range(17):
                _instance.keypoint[i, 2] = -1

            return result_bbox, _instance.keypoint

        # Try to extrapolate unseen instances but have detected while long time
        for idx, inst in enumerate(self._old_instances):
            if inst.extra and not inst.overlap and len(inst.path) > 40:
                inst.bbox, inst.keypoint = extrapolation_bbox(inst)
                inst.hide_time += 1
                inst.hide = True

        return [inst.color for inst in self._old_instances], [
            inst.index for inst in self._old_instances
        ]
