import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from slam_msgs.msg import DetectionArray
import message_filters
from cv_bridge import CvBridge
import numpy as np
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import open3d as o3d
import copy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import csv
import os

class ReconstructionNode(Node):
    def __init__(self):
        super().__init__('reconstruction_node')
        self.bridge = CvBridge()
        
        # Camera Intrinsics placeholders
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Subscribe to Camera Info to get the real pinhole intrinsics
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/cpr_r100_0000/sensors/camera_0/depth/camera_info',
            self.camera_info_callback,
            10)
            
       # Time-Synchronized Subscribers for Detections and Depth
        # NEW: Added qos_profile_sensor_data to explicitly catch Gazebo's Best Effort depth streams
        self.det_sub = message_filters.Subscriber(self, DetectionArray, '/tracking/detections')
        self.depth_sub = message_filters.Subscriber(self, Image, '/cpr_r100_0000/sensors/camera_0/depth/image', qos_profile=qos_profile_sensor_data)

        # NEW: Increased slop from 0.1 to 0.5 seconds to account for YOLO GPU processing latency
        self.ts = message_filters.ApproximateTimeSynchronizer([self.det_sub, self.depth_sub], queue_size=100, slop=0.5)
        self.ts.registerCallback(self.sync_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- NEW: Point Cloud Memory Bank ---
        self.object_clouds = {}  # Stores {tracking_id: o3d.geometry.PointCloud()}
        self.object_history = {} # Stores {tracking_id: {'last_center': np.array, 'last_time': float}}
        self.max_points = 5000   # Memory roof
        self.voxel_size = 0.05   # 5cm grid cubes
        self.velocity_threshold = 0.15

        self.csv_path = os.path.expanduser('/home/diksha/slam_ws/src/data_logging/warehouse_mapping_log.csv')
        
        # Initialize file and write headers if it doesn't exist yet
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Object_ID', 'Centroid_X', 'Centroid_Y', 
                    'Centroid_Z', 'Velocity_ms', 'Is_Static', 'Point_Count'
                ])

        # --- NEW: RViz Network Publishers ---
        self.pc_pub = self.create_publisher(PointCloud2, 'mapped_objects_cloud', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'mapped_objects_centroids', 10)
        # -------------------------------------
        
        self.get_logger().info("3D Reconstruction Node Online. Awaiting synchronized frames.")

    def camera_info_callback(self, msg):
        # Extract intrinsic matrix values once
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(f"Intrinsics Locked: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def sync_callback(self, det_msg, depth_msg):
        if self.fx is None:
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            # Clean the source frame for TF2
            safe_frame = depth_msg.header.frame_id
            if safe_frame.startswith('/'): safe_frame = safe_frame[1:]
            if safe_frame == 'camera_0_left_camera_optical_frame': 
                safe_frame = 'camera_0_left_camera_frame_optical'

            # Get the robot's current location in the world
            try:
                t = self.tf_buffer.lookup_transform(
                    'odom', safe_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1)
                )
            except Exception as tf_e:
                self.get_logger().warn(f"TF2 Transform Failed: {tf_e}")
                return # Skip frame if we don't know where the robot is

            for det in det_msg.detections:
                track_id = det.tracking_id
                
                # 1. Bounding Box Pixel Coordinates
                ymin, ymax = int(det.ymin), int(det.ymax)
                xmin, xmax = int(det.xmin), int(det.xmax)
                
                # Slicing the depth array (NumPy optimization)
                box_depth = depth_image[ymin:ymax, xmin:xmax].astype(float)
                if box_depth.size == 0: continue
                
                # Convert mm to meters if needed
                box_depth = np.where(box_depth > 100.0, box_depth / 1000.0, box_depth)
                
                # Filter out invalid depth (0.0 or NaN)
                valid_mask = (box_depth > 0.0) & (~np.isnan(box_depth))
                if not np.any(valid_mask): continue
                
                # 2. Vectorized Pinhole Projection (Fast 3D Math)
                v, u = np.indices(box_depth.shape)
                v += ymin
                u += xmin
                
                z = box_depth[valid_mask]
                x = (u[valid_mask] - self.cx) * z / self.fx
                y = (v[valid_mask] - self.cy) * z / self.fy
                
                # Create a local Open3D point cloud
                local_points = np.column_stack((x, y, z))
                new_cloud = o3d.geometry.PointCloud()
                new_cloud.points = o3d.utility.Vector3dVector(local_points)
                
                # 3. Apply the TF2 Global Rotation & Translation Matrix
                q = t.transform.rotation
                trans = t.transform.translation
                R = o3d.geometry.get_rotation_matrix_from_quaternion([q.w, q.x, q.y, q.z])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [trans.x, trans.y, trans.z]
                
                new_cloud.transform(T)

                # --- NEW: Calculate Timestamp & Current Centroid ---
                current_time = depth_msg.header.stamp.sec + (depth_msg.header.stamp.nanosec * 1e-9)
                current_centroid = np.asarray(new_cloud.get_center())
                is_static = True
                
                # 4. Voxel Grid Downsampling 
                new_cloud = new_cloud.voxel_down_sample(voxel_size=self.voxel_size)
                
                if track_id not in self.object_clouds:
                    # First time seeing this object! Initialize its history.
                    self.object_clouds[track_id] = new_cloud
                    self.object_history[track_id] = {
                        'last_center': current_centroid, 
                        'last_time': current_time
                    }
                    self.get_logger().info(f"Initialized Object {track_id} with {len(new_cloud.points)} points.")
                else:
                    # --- NEW: Velocity Evaluation Logic ---
                    prev = self.object_history[track_id]
                    dist = np.linalg.norm(current_centroid - prev['last_center'])
                    dt = current_time - prev['last_time']
                    
                    velocity = 0.0
                    # FIX: Wait at least 200ms to smooth out bounding box jitter
                    if dt > 0.2: 
                        velocity = dist / dt
                        if velocity > self.velocity_threshold:
                            is_static = False
                            
                        # Only update the history clock when we actually measure
                        self.object_history[track_id] = {
                            'last_center': current_centroid, 
                            'last_time': current_time
                        }

                    if not is_static:
                        # DYNAMIC OBJECT: Do not perform ICP. 
                        # Overwrite the cloud so only the latest frame exists.
                        self.object_clouds[track_id] = new_cloud
                        self.object_history[track_id] = {
                            'last_center': current_centroid, 
                            'last_time': current_time
                        }
                        self.get_logger().warn(
                            f"Object {track_id} is DYNAMIC (v={velocity:.2f}m/s). Freezing map append."
                        )
                    else:
                        # STATIC OBJECT: Align new frame to history
                        history_cloud = self.object_clouds[track_id]
                        
                        icp_result = o3d.pipelines.registration.registration_icp(
                            new_cloud, history_cloud, max_correspondence_distance=0.1,
                            init=np.eye(4),
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                        )
                        
                        new_cloud.transform(icp_result.transformation)
                        
                        merged_cloud = history_cloud + new_cloud
                        merged_cloud = merged_cloud.voxel_down_sample(voxel_size=self.voxel_size)
                        
                        points_arr = np.asarray(merged_cloud.points)
                        if len(points_arr) > self.max_points:
                            indices = np.random.choice(len(points_arr), self.max_points, replace=False)
                            merged_cloud.points = o3d.utility.Vector3dVector(points_arr[indices])
                        
                        self.object_clouds[track_id] = merged_cloud
                        self.object_history[track_id] = {
                            'last_center': np.asarray(merged_cloud.get_center()), 
                            'last_time': current_time
                        }
                        self.get_logger().info(
                            f"Box {track_id} Updated: {len(merged_cloud.points)}/{self.max_points} points (ICP Fitness: {icp_result.fitness:.2f})"
                        )
                        self.publish_visualizations(depth_msg.header.stamp)

                        # --- CSV LOG ROW PREPARATION ---
                    num_points = len(self.object_clouds[track_id].points)
                        
                    with open(self.csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            round(current_time, 2),
                            track_id,
                            round(float(current_centroid[0]), 3),
                            round(float(current_centroid[1]), 3),
                            round(float(current_centroid[2]), 3),
                            round(float(velocity), 2),
                            int(is_static),
                             num_points
                        ])
        except Exception as e:
            self.get_logger().error(f"Reconstruction failed: {e}")

    def publish_visualizations(self, stamp):
        if not self.object_clouds:
            return

        global_cloud = o3d.geometry.PointCloud()
        marker_array = MarkerArray()

        for track_id, cloud in self.object_clouds.items():
            # Merge into a single master cloud for RViz performance
            global_cloud += cloud
            
            # --- TASK 3.3: Calculate Statistical Centroid ---
            centroid = cloud.get_center()
            
            # --- TASK 3.4: Build the Centroid Marker ---
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = stamp
            marker.ns = 'centroids'
            marker.id = track_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # 20cm red sphere
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)

        # Broadcast the Markers
        self.marker_pub.publish(marker_array)

        # --- TASK 3.4: Convert and Broadcast Point Cloud ---
        points = np.asarray(global_cloud.points)
        if len(points) == 0:
            return
            
        header = Header()
        header.frame_id = 'odom'
        header.stamp = stamp
        
        # Fast NumPy to ROS 2 PointCloud2 conversion
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReconstructionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()