import numpy as np
import typing, logging
import motmetrics as mm

class EvaluatorOnline:
    '''
    在线生成评价指标。
    未完成。
    '''
    def __init__(self, max_age) -> None:
        '''
        max_age : int
            Maximum number of missed misses before a track is deleted. 
            只能获得 track_id 难以获得真正 track ，因此用 max_age 手动计算失配态是否变成删除态
        '''
        self._fp_cur = 0    # 越低越好。The total number of false positives. 错误预测的 track。 关注于当前帧
        self._fp_all = 0    #          关注于所有帧
        self._fn_cur = 0    # 越低越好。The total number of false negatives. 未被匹配的 ground truth
        self._fn_all = 0

        self._fm = 0        # 越低越好。The total number of times a trajectory is fragmented. 确定态track满足：tracked->untracked->tracked，FM 与 ID 是否发生变化无关
        # 设计如下4状态的有限状态机来维护该指标，保证了失配态只来自于匹配态，只要失配态进入匹配态那就是要记录的 fm
        # 初始态(新trk) -→ 匹配态(匹配trk) -→ 失配态(失配trk)
        #     |            ↑ |  ↑              |  ↑ |   |
        #     |            └-┘  └--[计入fm]----┘  └-┘   |
        #     └---------→ 删除态(删除trk) ←--------------┘
        self.__new_trks = set()         # 初始态，{ trk_id }
        self.__matched_trks = set()     # 匹配态，{ trk_id }
        self.__unmatched_trks = dict()  # 失配态，{ trk_id: time_since_update }      删除态直接删除，不用建表
        self.__max_age = max_age        # 用于手动计算失配态是否变成删除态

        ######## 下面是没有思路的指标
        # self._id_sw = 0     # 越低越好。The total number of identity switches. 所有跟踪目标身份交换的次数



    def update(self, det_id_2_track_id, matches, unmatched_tracks, unmatched_detections):
        self._fp_cur = len(unmatched_tracks)        # 错误预测的 track
        self._fp_all += self._fp_cur
        self._fn_cur = len(unmatched_detections)    # 未被匹配的 ground truth
        self._fn_all += self._fn_cur

        # 更新 fm: 按照 失配态、匹配态、初始态的顺序
        cur_matched_trks = [ma_trk_id for (ma_trk_id, ma_det_id) in matches]
        # 先更新失配态进入匹配态，再清理长期失配态，最后剩下没动的失配态时间+1
        for un_trk_id in list(self.__unmatched_trks):
            if un_trk_id in cur_matched_trks:         # 需要更新 fm 的关键位置：失配态进入匹配态 untracked->tracked
                del self.__unmatched_trks[un_trk_id]
                self.__matched_trks.add(un_trk_id)
                self._fm += 1
            elif self.__unmatched_trks[un_trk_id] > self.__max_age:
                del self.__unmatched_trks[un_trk_id]
            else:
                self.__unmatched_trks[un_trk_id] += 1
        # 更新匹配态进入失配态
        for ma_trk_id in list(self.__matched_trks):
            if ma_trk_id in unmatched_tracks:
                self.__matched_trks.remove(ma_trk_id)
                self.__unmatched_trks[ma_trk_id] = 1
        # 先清空老初始态，再迎接新初始态
        for trk_id in list(self.__new_trks):
            self.__new_trks.remove(trk_id)
            if trk_id in cur_matched_trks:
                self.__matched_trks.add(trk_id)
        for un_det_id in unmatched_detections:
            self.__new_trks.add(det_id_2_track_id[un_det_id])
        
class EvaluatorOffline:
    '''
    自己从检测的离线文件 track_result_file, ground_truth_file 中生成评价指标。
    未完成。
    '''
    def __init__(self, track_result_file, ground_truth_file) -> None:
        '''
        track_result_file、ground_truth_file: 按 MOT 格式保存的跟踪数据，每一行格式如下：
        frame_id, trail_id, x1, y1, w, h, conf, -1, -1, -1
        '''
        self.dets = np.loadtxt(track_result_file, delimiter=',')
        self.grtr = np.loadtxt(ground_truth_file, delimiter=',')

        self._fp_cur = 0    # 越低越好。The total number of false detections. 错误预测的 track。 关注于当前帧
        self._fp_all = 0    #          关注于所有帧
        self._fn_cur = 0    # 越低越好。The total number of missed detections. 未被匹配的 ground truth
        self._fn_all = 0

    def evaluate(self):
        frame_ids = set(self.grtr[:,0])
        for frame_id in frame_ids:
            frame_dets = self.dets[self.dets[:,0]==frame_id]
            frame_grtr = self.grtr[self.grtr[:,0]==frame_id]
            self.update(frame_dets, frame_grtr)

    def update(self, frame_dets, frame_grtr):
        '''
        frame_dets, frame_grtr: 同一帧的检测数据和真实数据
        '''
        # self._
        pass

class EvaluatorOfflineMotmetrics:
    '''
    修改自 motmetrics.apps.eval_motchallenge
    即 https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/apps/eval_motchallenge.py
    '''
    def __init__(self,
                gt_file, det_file,
                loglevel='info', fmt='mot15-2D',
                solver=None, id_solver=None,
                exclude_id=False
        ):
        '''
        parser.add_argument('gt_file', type=str, help='Path to ground truth file.') \n
        parser.add_argument('det_file', type=str, help='Path to tracker result file') \n
        parser.add_argument('--loglevel', type=str, help='Log level', default='info') \n
        parser.add_argument('--fmt', type=str, help='Data format',
            choices=['mot-16','mot15-2D','vatic_txt','detrac_mat','detrac_xml'], default='mot15-2D') \n
        parser.add_argument('--solver', type=str, help='LAP solver to use for matching between frames.') \n
        parser.add_argument('--id_solver', type=str, help='LAP solver to use for ID metrics. Defaults to --solver.') \n
        parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true',
                            help='Disable ID metrics') \n
        '''
        loglevel = getattr(logging, loglevel.upper(), None)
        if not isinstance(loglevel, int):
            raise ValueError('Invalid log level: {} '.format(loglevel))
        logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

        if solver:
            mm.lap.default_solver = solver
        
        logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
        logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
        logging.info('Loading files.')

        gt = mm.io.loadtxt(gt_file, fmt=fmt, min_confidence=1)
        det = mm.io.loadtxt(det_file, fmt=fmt)

        mh = mm.metrics.create()
        # Builds accumulator
        acc = mm.utils.compare_to_groundtruth(gt, det, 'iou', distth=0.5)

        metrics = list(mm.metrics.motchallenge_metrics)
        if exclude_id:
            metrics = [x for x in metrics if not x.startswith('id')]
        
        logging.info('Running metrics')

        if id_solver:
            mm.lap.default_solver = id_solver
        summary = mh.compute(acc)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        logging.info('Completed')


track_result_file = '/home/xxy/deep_sort/datasets/MOT15/train/ADL-Rundle-6/det/det.txt'
ground_truth_file = '/home/xxy/deep_sort/datasets/MOT15/train/ADL-Rundle-6/gt/gt.txt'
if __name__ == '__main__':
    # e = EvaluatorOffline(track_result_file, ground_truth_file)
    # e.evaluate()
    e = EvaluatorOfflineMotmetrics(ground_truth_file, track_result_file)