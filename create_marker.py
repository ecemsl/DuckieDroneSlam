import cv2
import numpy as np
import cv2.aruco as aruco
import os

# PART 1: GENERATE ARUCO MARKERS

def generate_aruco_markers(aruco_dictionary):
    # Generate 5 ArUCo tags (IDs 0 to 4) and save them to files
    for i in range(20):
        # Generate the ArUCo tag
        tag = np.zeros((300, 300, 1), dtype="uint8")
        cv2.aruco.generateImageMarker(aruco_dictionary, i, 300, tag, 1)

        # Add padding, without it detectMarkers cannot detect
        padding = 20
        tag_with_padding = cv2.copyMakeBorder(tag, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)

        # Save the tag as an image file
        output_path = os.path.join("aruco_tags_images", f"marker_{i}.png")
        cv2.imwrite(output_path, tag_with_padding)
        print(f"[INFO] Saved ArUCo tag {i} to {output_path}")

        # Display the tag on the screen
        cv2.imshow(f"ArUCo Tag {i}", tag)
        cv2.waitKey(500)  # Show each tag for 500 ms

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def generate_aruco_grid(aruco_dictionary):
    # Define the gridboard parameters
    x = 4
    y = 4
    markerLength = 0.04
    markerSeparation = 0.01
    board_size = (x, y)

    # Create the gridboard object using the constructor
    gridboard = aruco.GridBoard(
        size=board_size,
        markerLength=markerLength,
        markerSeparation=markerSeparation,
        dictionary=aruco_dictionary
    )

    # Create an image from the gridboard
    img = gridboard.generateImage(outSize=(600, 600), marginSize=5, borderBits=1)
    cv2.imwrite(f"aruco_tags_images/gridboard{x * y}.png", img)


# Create aruco marker gridboard
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
# generate_aruco_grid(aruco_dict)

# Generate aruco markers
generate_aruco_markers(aruco_dict)
