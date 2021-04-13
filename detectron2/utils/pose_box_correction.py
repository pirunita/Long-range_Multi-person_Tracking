# Keypoint correction part using information of adjacent fame
def keypoint_correction(frame_instances, window_size):

    keypoint_threshold = 0.03
    for inst in frame_instances[window_size]:
        match_instances = []
        for i, per_frame_instances in enumerate(frame_instances):
            for frame_inst in per_frame_instances:
                if inst.index == frame_inst.index:
                    match_instances.append(frame_inst)
        for j in range(17):
            # Using the keypoints which of score is larger than
            # keypoint threshold
            if inst.keypoint[j, 2] < keypoint_threshold and inst.keypoint[j, 2] > 0:
                plus_num = 0
                x_tune = 0
                y_tune = 0
                s_tune = 0
                for match_inst in match_instances:
                    if match_inst.keypoint[j, 2] > keypoint_threshold:
                        plus_num += 1
                        x_tune += match_inst.keypoint[j, 0]
                        y_tune += match_inst.keypoint[j, 1]
                        s_tune += match_inst.keypoint[j, 2]
                if plus_num > 0:
                    inst.keypoint[j, 0] = x_tune / plus_num
                    inst.keypoint[j, 1] = y_tune / plus_num
                    inst.keypoint[j, 2] = s_tune / plus_num

            # If corrected keypoint is out of bbox,
            # make score of that keypoimnt as zero
            if (
                inst.keypoint[j, 0] < inst.bbox[0]
                or inst.keypoint[j, 1] < inst.bbox[1]
                or inst.keypoint[j, 0] > inst.bbox[2]
                or inst.keypoint[j, 1] > inst.bbox[3]
            ):
                inst.keypoint[j, 2] = 0

    return [inst.keypoint.tolist() for inst in frame_instances[window_size]]
