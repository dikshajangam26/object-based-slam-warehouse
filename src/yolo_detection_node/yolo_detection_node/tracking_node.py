import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import csv
import os

from slam_msgs.msg import Detection, DetectionArray

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/cpr_r100_0000/sensors/camera_0/color/image',
            self.image_callback,
            10)
            
        # Updated output topics
        self.image_pub = self.create_publisher(Image, '/tracking/annotated_image', 10)
        self.bbox_pub = self.create_publisher(DetectionArray, '/tracking/detections', 10)
        
        self.bridge = CvBridge()
        
        # Load your custom trained model
        self.model = YOLO('/home/diksha/slam_ws/dataset/yolo_data/runs/detect/train-2/weights/best.pt')
        
        self.track_register = {}
        self.max_history = 30 # Number of past positions to remember

        self.get_logger().info("ByteTrack Temporal Engine Online. Assigning continuous IDs.")
        
        # --- CSV LOGGING CONFIGURATION ---
        self.csv_path = os.path.expanduser('/home/diksha/slam_ws/src/data_logging/yolo_tracking_log.csv')
        
        # Initialize file and write headers if it doesn't exist yet
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Track_ID', 'X_Min', 'Y_Min', 
                    'X_Max', 'Y_Max'
                ])

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run inference WITH the ByteTrack tracker activated
            results = self.model.track(cv_image, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # 1. Publish visual feed (Now includes IDs on the boxes)
            annotated_frame = results[0].plot()
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(ros_image)
            
            # 2. Extract mathematical coordinates and Temporal IDs
            det_array = DetectionArray()
            det_array.header = msg.header
            
            # Ensure the tracker successfully assigned IDs in this frame
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                # Get current ROS time in seconds for timestamping
                current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, class_ids):
                    # 1. Calculate the center point (centroid) of the box
                    cx = float((box[0] + box[2]) / 2.0)
                    cy = float((box[1] + box[3]) / 2.0)
                    class_name = self.model.names[cls_id]
                    
                    # 2. Update the Live Track Register
                    if track_id not in self.track_register:
                        # First time seeing this object! Create its profile.
                        self.track_register[track_id] = {
                            'class': class_name,
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'hit_count': 1,
                            'centroids': [(cx, cy)]
                        }
                    else:
                        # We know this object! Update its history.
                        self.track_register[track_id]['last_seen'] = current_time
                        self.track_register[track_id]['hit_count'] += 1
                        self.track_register[track_id]['centroids'].append((cx, cy))
                        
                        # Prevent memory leaks by capping the history list
                        if len(self.track_register[track_id]['centroids']) > self.max_history:
                            self.track_register[track_id]['centroids'].pop(0)

                    hits = self.track_register[track_id]['hit_count']
                    history_len = len(self.track_register[track_id]['centroids'])
                    self.get_logger().info(f"Box ID {track_id} | Hits: {hits} | Centroids Saved: {history_len}/{self.max_history}")
                    
                    # 3. Create and publish the standard ROS message
                    det = Detection()
                    det.xmin = int(box[0])
                    det.ymin = int(box[1])
                    det.xmax = int(box[2])
                    det.ymax = int(box[3])
                    det.confidence = float(conf)
                    det.class_name = class_name
                    det.tracking_id = track_id
                    
                    det_array.detections.append(det)
                    
                    # --- FIX: CSV LOG ROW PREPARATION (Moved inside the loop) ---
                    with open(self.csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            round(current_time, 2),
                            track_id, 
                            round(float(box[0]), 1), # X_Min
                            round(float(box[1]), 1), # Y_Min
                            round(float(box[2]), 1), # X_Max
                            round(float(box[3]), 1)  # Y_Max
                        ]) 
                
                self.bbox_pub.publish(det_array)

        except Exception as e:
            self.get_logger().error(f"Tracking failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()