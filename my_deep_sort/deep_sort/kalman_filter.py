# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)   # F
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)       # H

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance     # X_{0,0}, P_{0,0}

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # motion_cov == Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        # np.r_ 按列连接两个矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x' = Fx
        mean = np.dot(self._motion_mat, mean)
        # P' = FPF^T+Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance     # X_{k+1,k}, P_{k+1,k} 

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # innovation_cov == R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即 z=Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即 S=HP'H^T+R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov    # Z, S

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 将均值和协方差映射到检测空间，得到 x=Hx' 和 S=HP'H^T+R
        projected_mean, projected_cov = self.project(mean, covariance)

        
        # 计算卡尔曼增益 K=(P')(H.T)(S.-1)   S=HP'H.T+R
        # 为避免求逆（运算量大且数值不稳定），两侧右乘S得KS=P'H.T ，转变成 XA=B 的形式
        # 再将两边转置得 (S.T)(K.T)=(P'H.T).T ，获得 AX=B 的形式
        # AX=B 可通过 scipy.linalg.cho_solve 快速求解，传参为 (A 的 Cholesky分解因子) 和 (B)
        # 对A=S.T=S（协方差矩阵为对称阵）做Cholesky矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 传入A的Cholesky分解因子和B=(P'H.T).T，求解 AX=B   X=K.T   K=X.T
        kalman_gain = scipy.linalg.cho_solve(           # shape: (8, 4)
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx'
        innovation = measurement - projected_mean       # shape: (4,)
        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T) # mean.shape: (8,) np中 (8,4)*(4,)==(8,)==(4,)*(4,8)
        # P = (I - KH)P'  但代码里是 P=P'-KSK.T
        # 经https://zhuanlan.zhihu.com/p/497786053证明 KHP'==KSK.T
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance     # X_{k,k}, P_{k,k}

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        返回 mean(=x) 和 measurements(=z) 之间的马氏距离

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        #将mean和convariance映射到检测空间
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean # d=x-u
        # 马氏距离公式定义：DM = sqrt(d * inv(cov) * d.T)  即  DM^2 = d * inv(cov) * d.T
        # 但是按照定义直接计算 inv(cov) 过于笨重，因此下面使用 cholesky分解简化协方差求逆的运算
        #   cholesky分解的前提是：
        #   1.被分解矩阵需要是正定或半正定矩阵。（半正定分解的话，结果不唯一）
        #   2.被分解矩阵需要是对称矩阵/Hermitian矩阵。（前者是实数域，后者是复数域）
        #   协方差 cov 满足上述正定条件
        # cov = L * L.T   (cholesky分解)
        # 又cov是对称阵，故 L=L.T 也是对角阵
        # 则 DM^2 = d * (L * L.T).-1 * d.T
        #         = d * L.T.-1 * L.-1 * d.T
        #         = (L.-1 * d.T).T  *  (L.-1 * d.T)
        # 令 z = L.-1 * d.T
        # 则 DM^2 = z.T * z
        # 经验证，此解法能稳定跟踪，直接按定义写反而不能。。。。。
        # 按定义写： squared_maha = np.sum(d @ np.linalg.inv(covariance) @ d.T, axis=0)
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular( # 求z = L.-1 * d.T 即解 L*z=d.T，因L是对角阵，故可用此函数快速求解
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha # shape: (22,)


'''
扩展上面的原版官方卡尔曼包的关键位置：
__init__():
    F: self._motion_mat
    H: self._update_mat
initiate():
    X_{0,0}: return[0] == mean
    P_{0,0}: return[1] == covariance
predict():
    Q: motion_cov
    # X_{k+1,k}: return[0] == mean        # 这两个自动运算，不用改
    # P_{k+1,k}: return[1] == covariance
project():
    R: innovation_cov
    # Z: return[0] == mean                # 这两个自动运算，不用改
    # S: return[1] == covariance + innovation_cov
update():
    # 应该完全不用改
gating_distance():
    # 应该完全不用改
'''




class kalmanFilter_QR(KalmanFilter):
    """
    定制卡尔曼滤波器的 Q(motion_cov:预测过程中噪声协方差) 和 R(innovation_cov:测量过程中噪声的协方差)
    
    Parameters in args
    ----------
    Q_times : float >0
    R_times : float >0
    """
    def __init__(self, args):
        '''
        预测模型和原版保持一致，仅扩展Q和R。 \n
        8维状态空间： [x, y, a, h,    x', y', a', h'].T  \n
        4维观测空间： [x, y, a, h].T  \n
        Parameters in args
        ----------
        Q_times : 超参数，用于调整 超参数motion_cov（过程噪声协方差） 的大小
        R_times : 超参数，用于调整 超参数innovation_cov（观测噪声协方差） 的大小
        '''
        self.ndim, dt = 4, 1.
        self.xdim, self.zdim = 8, 4

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(self.xdim, self.xdim)   # F
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.zdim, self.xdim)       # H

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        assert args.Q_times > 0 and args.R_times > 0
        self.Q_times = args.Q_times
        self.R_times = args.R_times

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # motion_cov == Q 预测过程中噪声协方差
        motion_cov = np.diag(np.random.random(self.xdim)) * self.Q_times

        # x' = Fx
        mean = np.dot(self._motion_mat, mean)
        # P' = FPF^T+Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # innovation_cov == R 测量过程中噪声的协方差
        innovation_cov = np.diag(np.random.random(self.zdim)) * self.R_times

        # 将均值向量映射到检测空间，即 Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即 HP'H^T+R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

class kalmanFilter_xy(kalmanFilter_QR):
    '''
    预测模型中对x和y的加速度建模
    10维状态空间： [x, y, a, h,    x', y', a', h',   x'', y''].T
     4维观测空间： [x, y, a, h].T
    '''
    def __init__(self, args):
        '''
        预测模型中对x和y的加速度建模，并扩展Q和R。 \n
        10维状态空间： [x, y, a, h,    x', y', a', h',   x'', y''].T \n
        4维观测空间： [x, y, a, h].T \n
        Parameters in args
        ----------
        Q_times : 超参数，用于调整 超参数motion_cov（过程噪声协方差） 的大小
        R_times : 超参数，用于调整 超参数innovation_cov（观测噪声协方差） 的大小
        '''
        self.ndim, dt = 4, 1.
        self.xdim, self.zdim = 10, 4

        # Create Kalman filter model matrices.
        self._motion_mat = np.array([       # F: 10*10
            [1,0,0,0,   dt, 0, 0, 0,    0.5*dt*dt,         0],
            [0,1,0,0,    0,dt, 0, 0,    0,         0.5*dt*dt],
            [0,0,1,0,    0, 0,dt, 0,    0,                 0],
            [0,0,0,1,    0, 0, 0,dt,    0,                 0],
            
            [0,0,0,0,    1, 0, 0, 0,    dt,                0],
            [0,0,0,0,    0, 1, 0, 0,    0,                dt],
            [0,0,0,0,    0, 0, 1, 0,    0,                 0],
            [0,0,0,0,    0, 0, 0, 1,    0,                 0],
            
            [0,0,0,0,    0, 0, 0, 0,    1,                 0],
            [0,0,0,0,    0, 0, 0, 0,    0,                 1],
        ])
        self._update_mat = np.array([       # H: 4*10
            [1,0,0,0,    0, 0, 0, 0,    0,                 0],
            [0,1,0,0,    0, 0, 0, 0,    0,                 0],
            [0,0,1,0,    0, 0, 0, 0,    0,                 0],
            [0,0,0,1,    0, 0, 0, 0,    0,                 0]
        ])

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self._std_weight_acceleration = 1. / 3000
        assert args.Q_times > 0 and args.R_times > 0
        self.Q_times = args.Q_times
        self.R_times = args.R_times

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement              # x,y,a,h
        mean_vel = np.zeros_like(mean_pos)  # x',y',a',h'
        mean_acc = np.zeros(2)              # x'',y''
        mean = np.r_[mean_pos, mean_vel, mean_acc]    # X00

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],

            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
            
            80 * self._std_weight_acceleration * measurement[3],
            80 * self._std_weight_acceleration * measurement[3]]
        covariance = np.diag(np.square(std))        # P00
        return mean, covariance # X00, P00