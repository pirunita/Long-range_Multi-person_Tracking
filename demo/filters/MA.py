"""
Moving Average Filter (MAF)
"""

class MovingAvgFilter(object):
    def __init__(self, data, score_data, subset_size, avoid_fp_drift=True, mode='coord'):
        """
        Args: 
            data (list): 17 keypoints in each x and y coordinates per frames.
            score_data (list): pose estimation prediction score per frames. 
            subset_size (int): window size moving filter
            avoid_fp_drift: if True (the default) sums every sub-set rather than keeping a "rolling sum"
            (which may be subject to floating-point drift). While more correct, it is also dramatically 
            slower for subset sizes much larget than 20
        """
    
        if subset_size < 1:
            raise ValueError('subset_size must be 1 or larger')
        self.data = data
        self.score_data = score_data
        self.subset_size = subset_size
        self.mode = mode
        self.avoid_fp_drift = avoid_fp_drift
        self.divisor = float(subset_size)
    
    def predict(self):
        if self.avoid_fp_drift:
            if self.mode == 'score':
                for current in range(len(self.data)):
                    if self.data[current] == 0:
                        left_interpolate_values = []
                        right_interpolate_values = []
                        for val in range(1, self.subset_size+1):
                            left_filter = current - val
                            right_filter = current + val
                            if left_filter >= 0 and self.data[left_filter] > 0:
                                left_interpolate_values.append(self.data[left_filter])
                            if right_filter < len(self.data) and self.data[right_filter] > 0:
                                right_interpolate_values.append(self.data[right_filter])
                        if left_interpolate_values and right_interpolate_values:
                            yield (left_interpolate_values[0] + right_interpolate_values[0]) / 2
                        elif not left_interpolate_values and right_interpolate_values:
                            yield right_interpolate_values[0]
                        elif left_interpolate_values and not right_interpolate_values:
                            yield 0
                        elif not left_interpolate_values and not right_interpolate_values:
                            yield 0
                        
                    else:
                        left_interpolate_values = []
                        for val in range(1, self.subset_size+1):
                            left_filter = current - val
                            if left_filter >= 0 and self.data[left_filter] > 0:
                                left_interpolate_values.append(self.data[left_filter])
                        if left_interpolate_values:
                            yield sum(left_interpolate_values) / len(left_interpolate_values)
                        else:
                            yield self.data[current]
            else:
                for current in range(len(self.data)):
                    if self.score_data[current] == 0:
                        left_interpolate_values = []
                        right_interpolate_values = []
                        for val in range(1, self.subset_size+1):
                            left_filter = current - val
                            right_filter = current + val
                            if left_filter >= 0 and self.data[left_filter] > 0:
                                left_interpolate_values.append(self.data[left_filter])
                            if right_filter < len(self.data) and self.data[right_filter] > 0:
                                right_interpolate_values.append(self.data[right_filter])
                        
                        if left_interpolate_values and right_interpolate_values:
                            yield (left_interpolate_values[0] + right_interpolate_values[0]) / 2
                        elif not left_interpolate_values and right_interpolate_values:
                            yield right_interpolate_values[0]
                        elif left_interpolate_values and not right_interpolate_values:
                            yield 0
                        elif not left_interpolate_values and not right_interpolate_values:
                            yield 0
                    
                    else:
                        left_interpolate_values = []
                        for val in range(1, self.subset_size+1):
                            left_filter = current - val
                            if left_filter >= 0 and self.score_data[left_filter] > 0:
                                left_interpolate_values.append(self.data[left_filter])
                        if left_interpolate_values:
                            yield sum(left_interpolate_values) / len(left_interpolate_values)
                        else:
                            yield self.data[current]
                                    
    def interpolate(self, target):
        for i in range(len(target), len(self.data)):
            target.append(self.data[i])
        
        return target
        