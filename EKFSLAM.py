import cv2, cv2.aruco as aruco
import numpy as np
from collections import defaultdict
from scipy.linalg import block_diag

# ---------- Helper functions ----------------------------------------------------------

# Returns skew-symmetric matrix of a vector (used for rotation Jacobians)
def skew(v):
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,    -v[0]],
                     [-v[1], v[0],  0   ]])

# Converts Rodrigues vector to rotation matrix
def rodrigues_to_matrix(r):
    R, _ = cv2.Rodrigues(r)
    return R

# Converts rotation matrix to Rodrigues vector
def matrix_to_rodrigues(R):
    r, _ = cv2.Rodrigues(R)
    return r.reshape(3)

# Removes lens distortion from 2D points
def undistort_keypoints(pts, K, dist_coeffs):
    pts = np.expand_dims(pts, axis=1)  # shape (N, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undistorted.squeeze(1)

# ---------- EKF SLAM class ------------------------------------------------------------

class EKFSLAM:
    def __init__(self, Q_pose, R_meas):
        # Initial state: 6D pose (3 position + 3 orientation), no markers yet
        self.x  = np.zeros(6)            # State vector
        self.P  = np.eye(6) * 1e-3       # Covariance matrix
        self.Q  = Q_pose                 # Process noise for pose
        self.R  = R_meas                 # Measurement noise for each marker
        self.id2idx = {}                # Mapping from marker ID to index in state vector

    # -------- Prediction step (from visual odometry) ----------------------------------

    def predict(self, dR, dt):
        δθ  = matrix_to_rodrigues(dR)      # Convert relative rotation matrix to Rodrigues vector
        θ   = self.x[3:6]                  # Current orientation
        p   = self.x[0:3]                  # Current position
        R_w = rodrigues_to_matrix(θ)      # Rotation matrix from world to camera

        # Propagate position and orientation
        self.x[0:3] = p + R_w @ dt         # New position
        self.x[3:6] = θ + δθ               # New orientation

        # Build Jacobian F
        F_pose = np.eye(6)
        F_pose[0:3,3:6] = -R_w @ skew(dt)  # How orientation affects position
        F = block_diag(F_pose, np.eye(len(self.x)-6))

        # Propagate uncertainty
        self.P = F @ self.P @ F.T + block_diag(self.Q, np.zeros((len(self.x)-6, len(self.x)-6)))

    # -------- Update step (for observed marker) ----------------------------------------

    def update_marker(self, marker_id, z):
        if marker_id not in self.id2idx:
            # First time seeing this marker: initialize its 3D position
            idx = len(self.x)
            self.id2idx[marker_id] = idx
            p = self.x[0:3]
            θ = self.x[3:6]
            R_wc = rodrigues_to_matrix(θ)
            m_world = p + R_wc @ z  # Marker position in world coordinates
            #m_world[2] = 0
            self.x = np.hstack([self.x, m_world])
            self.P = block_diag(self.P, np.eye(3) * 1e2)  # High uncertainty for new marker

        # Get current estimates
        idx = self.id2idx[marker_id]
        p   = self.x[0:3]
        θ   = self.x[3:6]
        m   = self.x[idx:idx+3]
        R_cw = rodrigues_to_matrix(θ).T  # World to camera rotation

        # Expected marker position in camera frame
        z_hat = R_cw @ (m - p)

        # Build Jacobian H
        H_p   = -R_cw
        H_θ   = R_cw @ skew(m - p)
        H_m   = R_cw
        cols  = len(self.x)
        H = np.zeros((3, cols))
        H[:,0:3] = H_p
        H[:,3:6] = H_θ
        H[:,idx:idx+3] = H_m

        # Kalman update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        r = z - z_hat                          # Innovation
        self.x += K @ r                        # State correction
        self.P = (np.eye(cols) - K @ H) @ self.P  # Covariance correction

        print(f'[DEBUG] Marker {marker_id} z (cam frame): {z}')
        print(f'[DEBUG] Estimated marker {marker_id} world pos: {self.x[idx:idx + 3]}')
        print(f'[DEBUG] Camera position: {self.x[0:3]}, orientation (deg): {np.rad2deg(self.x[3:6])}')


# ------------------------------------------------------------------------------

# --- Main processing loop -----------------------------------------------------

video_path = "vid5.mp4"
cap = cv2.VideoCapture(video_path)

# Camera intrinsic matrix
K = np.array([
    [1658.70, 0, 788.73],
    [0, 1725.40, 740.24],
    [0, 0, 1]
])

# SIFT feature extractor and matcher
sift = cv2.SIFT_create()
bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# ArUco marker setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
aruco_det  = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

# EKF initialization
Q_pose = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
R_meas = np.eye(3)*0.01
ekf    = EKFSLAM(Q_pose, R_meas)

# Read first frame
ret, prev = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame from the video.")
prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = sift.detectAndCompute(prev_g, None)

trajectory = [ekf.x[0:3].copy()]  # Store camera positions

# --- Loop over video frames --------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret: break
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Feature matching for visual odometry
    kp, des = sift.detectAndCompute(g, None)
    if des_prev is None or des is None: continue
    matches = bf.match(des_prev, des)
    if len(matches) < 8: continue
    pts1 = np.float32([kp_prev[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp[m.trainIdx].pt   for m in matches])

    # Undistort keypoints
    dist_coeffs = np.array([-0.1093, 1.5111, -0.0705, -0.0008, -3.3128])
    pts1_undist = undistort_keypoints(pts1, K, dist_coeffs)
    pts2_undist = undistort_keypoints(pts2, K, dist_coeffs)

    # Estimate camera motion (Essential matrix)
    E, _ = cv2.findEssentialMat(pts1_undist, pts2_undist, K, cv2.RANSAC, 0.999, 1.0)
    if E is None: continue
    _, R, t, _ = cv2.recoverPose(E, pts1_undist, pts2_undist, K)

    # Predict new pose using VO
    ekf.predict(dR=R, dt=t.squeeze())

    # Detect markers and update EKF
    corners, ids, _ = aruco_det.detectMarkers(g)
    if ids is not None:
        for c, id_ in zip(corners, ids.flatten()):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c, 0.07, K, dist_coeffs)
            id_int = int(id_)

            if id_int > 50:
                continue


            is_loop_closure = id_int in ekf.id2idx
            prev_pose = ekf.x[0:3].copy()
            ekf.update_marker(id_int, tvec[0, 0])
            pose_change = np.linalg.norm(ekf.x[0:3] - prev_pose)
            print(f'[Loop Closure] Pose change from marker {id_int}: {pose_change:.4f}')

            if is_loop_closure:
                # Diagnostic: Compare predicted vs actual measurement
                idx = ekf.id2idx[id_int]
                p = ekf.x[0:3]
                θ = ekf.x[3:6]
                m = ekf.x[idx:idx + 3]
                R_cw = rodrigues_to_matrix(θ).T
                z_hat = R_cw @ (m - p)

                observed_dist = np.linalg.norm(tvec[0, 0])
                expected_dist = np.linalg.norm(z_hat)
                if expected_dist > 1e-3:
                    scale = observed_dist / expected_dist
                    print(f'[Scale Diagnostic] Marker {id_int} | obs: {observed_dist:.3f}, pred: {expected_dist:.3f}, scale: {scale:.3f}')

            #print('θ (Euler):', np.rad2deg(ekf.x[3:6]))  # Orientation in degrees

    trajectory.append(ekf.x[0:3].copy())
    kp_prev, des_prev = kp, des

cap.release()

# ------------- Visualize trajectory ------------------------------------------

trajectory = np.array(trajectory)*0.8

# Smooth trajectory with moving average
smoothed = []
window_size = 50
for i in range(len(trajectory)):
    start = max(0, i - window_size)
    end = min(len(trajectory), i + window_size + 1)
    avg_pose = np.mean(trajectory[start:end], axis=0)
    smoothed.append(avg_pose)

smoothed_traj = np.array(smoothed)

# Plot 3D trajectory and markers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D


# --- Final Static Plot ---
fig_static = plt.figure()
ax_static = fig_static.add_subplot(projection='3d')

# Plot full trajectory
ax_static.plot(smoothed_traj[:,0], smoothed_traj[:,1], smoothed_traj[:,2], 'b-', label='EKF trajectory')

# Plot marker positions
for id_, idx in ekf.id2idx.items():
    x, y, _ = ekf.x[idx:idx+3]
    ax_static.scatter(x, y, 0, s=40, label=f'marker {id_}')

ax_static.set_xlabel('X')
ax_static.set_ylabel('Y')
ax_static.set_zlabel('Z')
ax_static.legend()
plt.title('Trajectory with Marker Positions')
plt.show()


'''
# --- Animated Plot of Trajectory Build ---
fig_anim = plt.figure()
ax_anim = fig_anim.add_subplot(projection='3d')

# Empty line to be updated
line, = ax_anim.plot([], [], [], 'b-', label='EKF trajectory')

# Set consistent axes limits
ax_anim.set_xlim(np.min(smoothed_traj[:, 0]), np.max(smoothed_traj[:, 0]))
ax_anim.set_ylim(np.min(smoothed_traj[:, 1]), np.max(smoothed_traj[:, 1]))
ax_anim.set_zlim(np.min(smoothed_traj[:, 2]), np.max(smoothed_traj[:, 2]))
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_zlabel('Z')

# Static markers
for id_, idx in ekf.id2idx.items():
    x, y, _ = ekf.x[idx:idx+3]
    ax_anim.scatter(x, y, 0, s=40, label=f'marker {id_}')
ax_anim.legend()

# Animation update function
def update(frame):
    line.set_data(smoothed_traj[:frame, 0], smoothed_traj[:frame, 1])
    line.set_3d_properties(smoothed_traj[:frame, 2])
    return line,

# Animate
ani = FuncAnimation(fig_anim, update, frames=len(smoothed_traj), interval=20, blit=False)

# Save to MP4
print("saving animation")
writer = FFMpegWriter(fps=30, metadata=dict(artist='eco'), bitrate=1800)
ani.save('trajectory_animation.mp4', writer=writer)

plt.close(fig_anim)  # Close animation window after saving
'''




