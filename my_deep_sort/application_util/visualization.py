# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
try:
    from .image_viewer import ImageViewer
except:
    from image_viewer import ImageViewer
import os, cv2, random

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    # def __init__(self, seq_info):
    #     self.frame_idx = seq_info["min_frame_idx"]
    #     self.last_idx = seq_info["max_frame_idx"]
    def __init__(self, first_idx, last_idx):
        self.frame_idx = first_idx
        self.last_idx = last_idx

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]              # (640, 480)
        aspect_ratio = float(image_shape[1]) / image_shape[0]   # 0.75
        image_shape = 1024, int(aspect_ratio * 1024)            # (1024, 768)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#

class Visualization_only_save_video(Visualization):
    ''' 未完成，输出的视频无法播放 '''
    def __init__(self, update_ms, output_video_path, first_idx, last_idx, image_shape):
    # def __init__(self, seq_info, update_ms):
        image_shape = image_shape[::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, '')
        self.viewer.thickness = 2
        self.frame_idx = first_idx
        self.last_idx = last_idx

        self.viewer.enable_videowriter(output_video_path)

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback), show=False)

class Visualization_only_save_image(Visualization):
    def __init__(self, output_image_folder, first_idx, last_idx, image_shape, image_names):
    # def __init__(self, seq_info, update_ms):
        image_shape = image_shape[::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.frame_idx = first_idx
        self.last_idx = last_idx
        self.image_names = image_names
        self.viewer = ImageViewer(
            0, image_shape, '', output_image_folder, first_idx, self.image_names)
        self.viewer.thickness = 2

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback), show=False)

    def draw_trails(self, trails, cur_frame_id, colors=None):
        '''
        该函数逻辑颜色不匹配，待更新
        trails: { track_id:[[本track最近trail_len帧框的中心点],最近的frame_id] } \n
        cur_frame_id: 当前帧的id
        colors: [(r,g,b),(),..] for r,b,g in [0,255], default is random
        '''
        if colors is None:
            colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(len(trails))]
        for trail in list(trails.values()):
            if trail[1] != cur_frame_id:
                continue
            for i,(x,y) in enumerate(trail[0]):
                self.viewer.circle(x,y,2)
                if i>0:
                    self.viewer.color = colors[i]
                    self.viewer.line(*trail[0][i],*trail[0][i-1])

    def draw_detections(self, detections, labels=None, colors=None):
        '''
        colors: [(r,g,b),(),..] for r,b,g in [0,255], default is random
        '''
        self.viewer.thickness = 2
        if colors is None:
            colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(len(detections))]
        if labels is None:
            for i, detection in enumerate(detections):
                self.viewer.color = colors[i]
                self.viewer.rectangle(*detection.tlwh)
        else:
            for i, detection in enumerate(detections):
                self.viewer.color = colors[i]
                self.viewer.rectangle(*detection.tlwh, labels[i])

    def draw_detection_masks(self, masks_bool, colors=None, labels=None):
        '''
        masks_bool: [np.array(..),np.array()，..]  np.array 内部是bool型
        colors: [(r,g,b),(),..] for r,b,g in [0,255], default is random
        '''
        if colors is None:
            colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(len(masks_bool))]
        for i,mask_bool in enumerate(masks_bool):                                   #     mask_bool.shape: (1080, 1920)
            self.viewer.image[mask_bool] = colors[i]  # 颜色与其他draw的框和trail的颜色保持一致 self.viewer.image.shape: (1080, 1920, 3)
        if labels is not None:
            for i,mask_bool in enumerate(masks_bool):
                cx,cy = _get_centroid(mask_bool.astype(np.uint8))
                if cx>-1 and cy>-1:
                    self.viewer.annotate(cx,cy,labels[i])

    def draw_trackers(self, tracks, track_colors):
        '''
        track_colors: { track_id: (r,g,b) } 用于统一条轨迹和其框的颜色
        '''
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = track_colors[track.track_id]
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)

def _get_centroid(img_gray):
    '''
    获取二值图像的最大轮廓的质心坐标
    参考:
    https://blog.csdn.net/LuohenYJ/article/details/88599334
    https://blog.csdn.net/qq_39197555/article/details/108997933
    http://edu.pointborn.com/article/2021/11/19/1709.html
    '''
    ret, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 由于分割图像破碎，可能有多个轮廓，因此计算次大轮廓的质心（最大通常不是）
    areas = [cv2.contourArea(c) for c in contours] # 第0个是整张图
    if len(areas)<2: # 无轮廓
        # cv2.imwrite('tmp/len_areas_small.jpg',img_gray)
        return -1,-1
    the_contour = contours[areas.index(sorted(areas)[-2])]
    M = cv2.moments(the_contour, binaryImage=True) # 计算该轮廓的矩
    if M["m00"]==0: # 无轮廓
        # cv2.imwrite('tmp/m00_zero.jpg',img_gray)
        return -1,-1
    cX = int(M["m10"] / M["m00"]) # 计算重心
    cY = int(M["m01"] / M["m00"])
    return cX, cY



class Image_saver(Visualization):
    def __init__(self, img, output_folder):
        ''' img：暂时只支持图片路径 '''
        img = cv2.imread(img)                                   # shape: (1080, 1920, 3)
        image_shape = img.shape[:2]                             # (1080, 1920)
        image_names = ['test.jpg']
        
        self.viewer = ImageViewer(
            500, image_shape, 'test fig', output_folder, 0, image_names)
        self.viewer.thickness = 2

        self.set_image(img.copy())
        fake_detections=[[614., 357., 779., 450.], [772.,  16., 946., 163.]]
        self.draw_detections(fake_detections)
        self.viewer.save_image()

    def draw_detections(self, tlwh_list):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, tlwh in enumerate(tlwh_list):
            self.viewer.rectangle(*tlwh)
    
    # def draw_trails(self, tlwh_list):


img_path = '/home/xxy/deep_sort/datasets/lingshui/30fps300s/0001.jpg'
test_out_folder = '/home/xxy/deep_sort/tmp'
maskpath = '/home/xxy/deep_sort/datasets/chores/mask.jpg'
if __name__ == '__main__':
    # i = Image_saver(img_path, test_out_folder)
    img = cv2.imread(maskpath)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[1]
    img_color1 = np.copy(img)
    img_color2 = np.copy(img)

    cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)

    area = cv2.contourArea(cnt)
    print('area: ', area)
    perimeter = cv2.arcLength(cnt, True)
    print('perimeter: ', perimeter)

    M = cv2.moments(cnt)
    print('M: ', M)
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    print('cx: ', cx, 'cy: ', cy)
    cv2.imwrite('tmp/tmp.jpg',img_color1)
    