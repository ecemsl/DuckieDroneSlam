# SLAM for Indoor Localization and Mapping on DuckieDrone
**Contributors: Ecem Selin Demir, Ahmet Tunca Taşkın, Pınar Bektaş, Supervisor: Asst. Prof. Dr. Özgür Erkent**

**WebSite of the project: https://dncaa.github.io/ArUco-SLAM-Webpage/**

---
##  About the Project

This project focuses on the development of a **GPS-free drone localization and mapping system** using **ArUco Tags** and **SLAM techniques**. In environments where GPS is unreliable or unavailable such as indoors, underground, or in disaster-stricken areas, accurate and real-time localization remains a significant challenge.

To address this, our system integrates **camera-based ARTag detection** with **visual-inertial SLAM algorithms**. The goal is to enable **real-time, accurate drone localization and mapping** using only onboard sensors.

---


##  What is EKF-SLAM?
EKF-SLAM (Extended Kalman Filter Simultaneous Localization and Mapping) is a technique where a robot (or drone) estimates its own position and the locations of landmarks (e.g., ArUco markers) simultaneously. It fuses noisy sensor data (like camera movement and visual landmarks) to continuously improve its estimate of the environment and itself.

This project uses visual-inertial odometry and ArUco marker observations to estimate the drone’s 3D pose and build a map of marker locations. It follows this EKF pipeline:

1. Prediction Step

Inputs: Drone motion from Visual Odometry (estimated with SIFT + Essential Matrix)

Goal: Predict where the drone is now

This updates the estimated drone pose using relative rotation dR and translation dt and the uncertainty of the estimate using Jacobians.

2. Correction Step

Inputs: Observed ArUco marker from the camera

Goal: Correct the drone’s pose and the marker’s position


## Preprocessing: Camera Calibration & Marker Detection
### Camera Calibration
To ensure accurate marker detection and pose estimation, we first calibrate the drone’s camera using a set of chessboard images. Calibration is needed to find correct lens distortion coefficients and intrinsic camera matrix.

This step is crucial for accurate 3D position estimation of ArUco markers and undistorting video frames before processing.

### ArUco Marker Detection
Once calibrated, we use ArUco markers as visual landmarks to help the drone localize itself.  ArUco markers are unique black-and-white square patterns
which are easily detectable and trackable.
They are used to estimate the camera's 6-DoF pose (position + orientation).


---

## How To Use DuckieDrone21?
This project runs on the DuckieDrone21 platform using a Docker-based ROS Kinetic workspace. Below are the essential steps:

A docker workspace is set up from the source code of the repository pidrone_pkg.


Drone flying process starts by checking the drone’s battery to ensure sufficient power. 

After connecting to drone's WI-FI, an SSH connection is established with the drone using _ssh duckie@amelia.local_ with the password _quackquack_.

The Docker container is started by navigating to the pidrone_pkg directory and running _rake start_. Then, a screen session is opened using _screen -c pi.screenrc_ to initialize the necessary ROS nodes and access the web interface to control the drone.

The Web Interface allows us to control the drone from the base station. To access it, we need to open _index.html_ in _/pidrone_pkg/web/_. To connect, we use the drone’s VLAN IP address as hostname.

Then we start up the Flight Controller node by going to the 1*$FC screen and run the command python flight_controller_node.py.


### Required Files & Sources
We used the following files and sources:

- **pidrone_pkg**: Main ROS package cloned from GitHub

- **DuckieTown Operation Manual**: It was used for hardware building details

### Custom calibration files:

- **camera_matrix.txt & distortion_coefficients.txt**: Computated by running **camera_calibration.py** script in this repository. 


### Important Notes
Particular attention was paid to:

Wire Management: Ensured no wires interfere with propeller movement; zip ties were used where needed.

Battery Safety: Battery was kept unplugged during setup. The battery used is Lithium-ion polymer (LiPo) battery with three cells, each holds 4.2 volt when charged, and 3.7 when discharged. **The battery should not be discharged below 10.5 V**. Also, for the drone to work properly, the voltage shouldn’t be below 11.3 V. 

