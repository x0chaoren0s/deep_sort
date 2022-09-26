# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    args : parser.parse_args()

    Parameters in args
    ----------
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    kalmanFilter_type : str in ['raw', 'ana_solu']
        default='raw'

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, args, max_iou_distance=0.7):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = args.max_age  # Maximum number of missed misses before a track is deleted.
        self.n_init = args.n_init    # Number of frames that a track remains in initialization phase.

        if args.kalmanFilter_type == 'raw':
            self.kf = kalman_filter.KalmanFilter()
        elif args.kalmanFilter_type == 'xy':
            self.kf = kalman_filter.KalmanFilter_xy(args)
        elif args.kalmanFilter_type == 'qr':
            self.kf = kalman_filter.KalmanFilter_QR(args)
        elif args.kalmanFilter_type == 'xyqr':
            self.kf = kalman_filter.KalmanFilter_xy_QR(args)
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):   # detections: [ (tlwh,confidence,feature),(),.. ]
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection] == [ (tlwh,confidence,feature),(),.. ]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)    # 比如 matches: [(0, 1), (1, 0), (2, 3), (3, 2), (4, 6), (5, 5), (6, 4), (7, 7), (8, 8), (9, 9), (10, 11)]
                                        # unmatched_detections: [10]  unmatched_tracks: []   matche 左边是 track_idx ，右边是 detection_idx
        det_id_2_track_id = dict()  # det_id_2_track_id 第一部分来自于matches，第二部分来自于unmatched_detections
        for track_idx, detection_idx in matches:
            det_id_2_track_id[detection_idx] = track_idx        # det_id_2_track_id 的第一部分

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            det_id_2_track_id[detection_idx] = self._next_id    # det_id_2_track_id 的第二部分
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        return det_id_2_track_id, matches, unmatched_tracks, unmatched_detections  # det_id_2_track_id 第一部分来自于matches，第二部分来自于unmatched_detections

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            '''
            根据外观信息和马氏距离，计算卡尔曼滤波预测到的tracks和当前时刻检测到的
            detections的代价矩阵
            '''
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            #基于外观信息（比如cos距离，欧氏距离，IOU距离），计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)
            #基于马氏距离，过滤掉代价矩阵中的一些不合理的项，将其设置为一个较大的值
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix #将这个代价矩阵应用一下匈牙利算法，就能进行检测detections和跟踪tracks的匹配了

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        
        
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())   # detection.to_xyah(): (x1+w/2,y1+h/2,w/h,h)
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
