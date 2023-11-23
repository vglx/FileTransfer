import rospy
import cv2
import torch
import numpy as np
from pyzed import sl
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def main():
    # Initialize the ROS node
    rospy.init_node('yolo_zed_node')

    # Create a publisher for detected objects information
    pub = rospy.Publisher('detected_objects', String, queue_size=10)

    # Initialize ZED camera
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.CENTIMETER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        rospy.logerr("Error opening ZED camera. Exiting...")
        return

    runtime = sl.RuntimeParameters()
    mat_l = sl.Mat()
    mat_r = sl.Mat()
    depth_map = sl.Mat()

    # Load YOLOv5 model
    model = torch.hub.load('./', 'custom', './best.pt', source='local')
    bridge = CvBridge()

    # Loop images capture and perform object detection
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Obtain left and right images
            zed.retrieve_image(mat_l, sl.VIEW.LEFT)
            zed.retrieve_image(mat_r, sl.VIEW.RIGHT)

            # Input the left image into the YOLOv5 model for object detection
            img_l = cv2.cvtColor(mat_l.get_data(), cv2.COLOR_RGBA2RGB)
            results = model(img_l, size=640)

            # Obtain depth map
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            targets_info = []

            # Traverse detected targets and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)

                # Returns a sl.float4 object. This object contains the X, Y, Z coordinates of a 3D point and the depth value of the point
                # X-axis: from left to right. When you look at a ZED camera, the positive X direction is from the left to the right of the camera.
                # Y-axis: from bottom to top. The positive Y direction is from the bottom of the camera to the top.
                # Z axis: from inside to outside. The positive Z direction is the direction from the camera lens into the scene.
                # coord_val[0] or coord_val.x: X coordinate
                # coord_val[1] or coord_val.y: Y coordinate
                # coord_val[2] or coord_val.z: Z coordinate
                # coord_val[3] or coord_val.w: Depth value, or distance from the camera to the point.
                coord_val = depth_map.get_value(x_center, y_center)

                # x_coord = coord_val.x
                # y_coord = coord_val.y
                # z_coord = coord_val.z
                # w_coord = coord_val.w

                targets_info.append([coord_val[0], coord_val[1], coord_val[2], coord_val[3], int(cls)])

            # Publish detected objects information
            pub.publish(str(targets_info))

            rate.sleep()

    # Release resources and close windows
    zed.close()

if __name__ == "__main__":
    main()