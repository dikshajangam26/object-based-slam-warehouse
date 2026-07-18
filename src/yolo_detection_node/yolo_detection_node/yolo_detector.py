import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Import our new custom messages
from slam_msgs.msg import Detection, DetectionArray

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Subscriptions and Publishers
        self.subscription = self.create_subscription(
            Image,
            '/cpr_r100_0000/sensors/camera_0/color/image',
            self.image_callback,
            10)
            
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.bbox_pub = self.create_publisher(DetectionArray, '/yolo/detections', 10)
        
        self.bridge = CvBridge()
        
        # Load the model. Change 'yolov8n.pt' to 'best.pt' if you ever train custom weights
        self.model = YOLO('/home/diksha/slam_ws/dataset/yolo_data/runs/detect/train-2/weights/best.pt') 
        
        self.get_logger().info("Upgraded YOLOv8 Engine Online. Publishing DetectionArrays.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run inference (Sequential Processing Latency Target)
            results = self.model(cv_image, verbose=False)
            
            # 1. Publish the visual image for RViz
            annotated_frame = results[0].plot()
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(ros_image)
            
            # 2. Extract and publish the mathematical DetectionArray
            det_array = DetectionArray()
            det_array.header = msg.header # Maintain time synchronization
            
            for box in results[0].boxes:
                det = Detection()
                coords = box.xyxy[0].tolist()
                
                det.xmin = int(coords[0])
                det.ymin = int(coords[1])
                det.xmax = int(coords[2])
                det.ymax = int(coords[3])
                det.confidence = float(box.conf[0])
                det.class_name = self.model.names[int(box.cls[0])]
                
                det_array.detections.append(det)
                
            self.bbox_pub.publish(det_array)
            
        except Exception as e:
            self.get_logger().error(f"Detection failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()