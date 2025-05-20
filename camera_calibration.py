import numpy as np
import cv2
import glob
import os
import pickle

# Calibration settings
BOARD_DIMENSIONS = (5, 7)       # Chessboard dimensions: (columns, rows)
CELL_SIZE_CM = 3.1              # Square size in centimeters
IMAGES_DIR = 'calibration_images'
EXPORT_DIR = 'output'
EXPORT_UNDISTORTED = True       # Toggle for saving undistorted images

def perform_calibration():
    """
    Executes the camera calibration using chessboard images.

    Returns:
        rms_error: Root Mean Square (RMS) re-projection error
        cam_matrix: Intrinsic camera matrix
        dist_coeffs: Lens distortion coefficients
        rot_vecs: Rotation vectors for each view
        trans_vecs: Translation vectors for each view
    """
    # Set up known 3D reference points in the chessboard's coordinate space
    world_pts = np.zeros((BOARD_DIMENSIONS[0] * BOARD_DIMENSIONS[1], 3), np.float32)
    world_pts[:, :2] = np.mgrid[0:BOARD_DIMENSIONS[0], 0:BOARD_DIMENSIONS[1]].T.reshape(-1, 2)
    world_pts *= CELL_SIZE_CM

    object_points = []  # Real-world coordinates
    image_points = []   # Pixel coordinates

    # Load image files
    image_files = glob.glob(os.path.join(IMAGES_DIR, '*.jpeg'))
    if not image_files:
        print(f"No JPEG images found in {IMAGES_DIR}")
        return None, None, None, None, None

    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"{len(image_files)} images found for calibration.")

    for idx, filepath in enumerate(image_files):
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"Could not read image {filepath}. Skipping.")
            continue
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Locate corners in chessboard pattern
        found, corners = cv2.findChessboardCorners(grayscale, BOARD_DIMENSIONS, None)

        if found:
            object_points.append(world_pts)

            # Enhance accuracy of corner detection
            termination = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            precise_corners = cv2.cornerSubPix(grayscale, corners, (11, 11), (-1, -1), termination)
            image_points.append(precise_corners)

            cv2.drawChessboardCorners(frame, BOARD_DIMENSIONS, precise_corners, found)
            result_path = os.path.join(EXPORT_DIR, f'corners_{os.path.basename(filepath)}')
            cv2.imwrite(result_path, frame)

            print(f"[{idx+1}/{len(image_files)}] Found corners in {filepath}")
        else:
            print(f"[{idx+1}/{len(image_files)}] Failed to detect corners in {filepath}")

    if not object_points:
        print("No valid chessboard detections. Calibration aborted.")
        return None, None, None, None, None

    print("Running calibration...")

    rms_error, cam_matrix, dist_coeffs, rot_vecs, trans_vecs = cv2.calibrateCamera(
        object_points, image_points, grayscale.shape[::-1], None, None
    )

    calibration_output = {
        'camera_matrix': cam_matrix,
        'distortion_coefficients': dist_coeffs,
        'rotation_vectors': rot_vecs,
        'translation_vectors': trans_vecs,
        'reprojection_error': rms_error
    }

    with open(os.path.join(EXPORT_DIR, 'calibration_data.pkl'), 'wb') as file:
        pickle.dump(calibration_output, file)

    np.savetxt(os.path.join(EXPORT_DIR, 'camera_matrix.txt'), cam_matrix)
    np.savetxt(os.path.join(EXPORT_DIR, 'distortion_coefficients.txt'), dist_coeffs)

    print(f"Calibration done. RMS Error = {rms_error:.4f}")
    print(f"Data saved in {EXPORT_DIR}")

    return rms_error, cam_matrix, dist_coeffs, rot_vecs, trans_vecs


def fix_image_distortion(cam_matrix, dist_coeffs):
    """
    Undistort and save the images using computed calibration parameters.
    """
    if not EXPORT_UNDISTORTED:
        return

    valid_images = []
    for ext in ('*.jpeg', '*.jpg', '*.png'):
        valid_images.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))

    if not valid_images:
        print("No valid image files for undistortion.")
        return

    undistorted_output_path = os.path.join(EXPORT_DIR, 'undistorted')
    os.makedirs(undistorted_output_path, exist_ok=True)

    print(f"Saving undistorted images...")

    for idx, filepath in enumerate(valid_images):
        img = cv2.imread(filepath)
        if img is None:
            print(f"Failed to load image: {filepath}")
            continue

        height, width = img.shape[:2]
        optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (width, height), 1, (width, height))
        corrected = cv2.undistort(img, cam_matrix, dist_coeffs, None, optimal_matrix)

        x, y, w, h = roi
        corrected = corrected[y:y+h, x:x+w]

        save_path = os.path.join(undistorted_output_path, f'undistorted_{os.path.basename(filepath)}')
        cv2.imwrite(save_path, corrected)

        print(f"[{idx+1}/{len(valid_images)}] Saved: {save_path}")

    print(f"Undistorted images saved in {undistorted_output_path}")


def reprojection_error_stats(obj_pts, img_pts, cam_matrix, dist_coeffs, rvecs, tvecs):
    """
    Print reprojection error for each frame.
    """
    total = 0
    for i in range(len(obj_pts)):
        projected_pts, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], cam_matrix, dist_coeffs)
        error = cv2.norm(img_pts[i], projected_pts, cv2.NORM_L2) / len(projected_pts)
        total += error
        print(f"Reprojection error [image {i+1}]: {error:.4f}")

    avg_error = total / len(obj_pts)
    print(f"Average reprojection error: {avg_error:.4f}")
    return avg_error


def execute():
    """
    Entrypoint: perform camera calibration and optional image correction.
    """
    print(">> Starting calibration sequence...")
    rms, K, d, Rs, Ts = perform_calibration()
    if K is None:
        print(">> Calibration failed.")
        return

    fix_image_distortion(K, d)
    print(">> Calibration and undistortion completed.")

if __name__ == "__main__":
    execute()
