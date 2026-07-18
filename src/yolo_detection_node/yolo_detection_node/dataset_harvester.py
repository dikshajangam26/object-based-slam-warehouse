import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class DatasetHarvester(Node):
    def __init__(self):
        super().__init__('dataset_harvester')
        self.subscription = self.create_subscription(
            Image,
            '/cpr_r100_0000/sensors/camera_0/color/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        # The folder where your synthetic images will be saved
        self.save_dir = os.path.expanduser('~/slam_ws/dataset/raw_images/')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.image_count = 0
        self.last_save_time = time.time()
        self.save_interval = 2.0  # Seconds between snapshots

        self.get_logger().info("Snapshot Harvester Online. Driving robot will auto-collect data.")

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                filename = os.path.join(self.save_dir, f"warehouse_frame_{self.image_count:04d}.jpg")
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f"Saved: {filename}")
                self.image_count += 1
                self.last_save_time = current_time
            except Exception as e:
                self.get_logger().error(f"Failed to save: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DatasetHarvester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()