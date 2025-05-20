import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PART 2: DETECT ARUCO MARKERS IN VIDEO

# Start capturing from the default webcam
cap = cv2.VideoCapture('aruco_tag_videos/test1.mp4')
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Starting video stream. Press 'q' to quit.")

# Get the width and height of the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Set up the video writer to save the video
out = cv2.VideoWriter('aruco_tag_videos/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                      (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("[ERROR] Failed to grab frame")
        break
        # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set up dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters()

    # Create the new detector object
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray_frame)

    # Draw markers if detected
    if ids is not None and len(ids) > 0:
        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0].astype(int)  # Get the 4 corner points of the marker

                # Draw a custom polygon frame
                cv2.polylines(frame, [c], isClosed=True, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

                # Optionally draw circles at each corner
                for point in c:
                    cv2.circle(frame, tuple(point), radius=6, color=(255, 0, 0), thickness=-1)
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                cv2.putText(frame, f"ID: {ids[i]}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Camera intrinsic matrix (3x3)
    camera_matrix = np.array([[500, 0, frame_width / 2],
                              [0, 500, frame_height / 2],
                              [0, 0, 1]], dtype=np.float32)

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))  # Set to zero for simplicity, adjust if you have distortion coefficients

    # Draw the pose estimation of the marker
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.07, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05, 3)

    
    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Video Capture', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()


