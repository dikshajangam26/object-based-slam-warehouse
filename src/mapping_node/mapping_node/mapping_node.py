import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import math

# TF2 Libraries
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        
        # Format: { global_id: {'class': str, 'x': float, 'y': float, 'z': float, 'hits': int, 'last_seen': float} }
        self.global_object_db = {}
        self.next_global_id = 1
        self.spatial_merge_threshold = 0.4 # 40cm Euclidean distance to trigger deduplication
        
        # TF2 Listeners
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.world_frame = 'odom'
        
        # Subscriptions & Publishers
        self.centroid_sub = self.create_subscription(
            MarkerArray,
            'mapped_objects_centroids',
            self.centroid_callback,
            10
        )
        
        self.global_map_pub = self.create_publisher(MarkerArray, 'global_semantic_map', 10)
        self.get_logger().info("Global Object Mapping Node Online. TF2 Listener Active (Frame: odom).")

    def apply_tf_transform(self, x, y, z, trans, rot):
        q_w, q_x, q_y, q_z = rot.w, rot.x, rot.y, rot.z
        R = np.array([
            [1 - 2*(q_y**2 + q_z**2),     2*(q_x*q_y - q_z*q_w),     2*(q_x*q_z + q_y*q_w)],
            [    2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2),     2*(q_y*q_z - q_x*q_w)],
            [    2*(q_x*q_z - q_y*q_w),     2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
        ])
        point_local = np.array([x, y, z])
        point_world = np.dot(R, point_local) + np.array([trans.x, trans.y, trans.z])
        return point_world[0], point_world[1], point_world[2]

    def centroid_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        db_updated = False
        
        for marker in msg.markers:
            source_frame = marker.header.frame_id
            lx, ly, lz = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
            
            # --- CRITICAL FIX: Ignore empty depth data (NaN or Inf) ---
            if any(math.isnan(val) or math.isinf(val) for val in (lx, ly, lz)):
                continue
                
            class_name = marker.text if marker.text else "object"
            wx, wy, wz = lx, ly, lz
            
            if source_frame != self.world_frame:
                try:
                    t = self.tf_buffer.lookup_transform(
                        self.world_frame,
                        source_frame,
                        marker.header.stamp
                    )
                    wx, wy, wz = self.apply_tf_transform(lx, ly, lz, t.transform.translation, t.transform.rotation)
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self.get_logger().warn(f"TF2 Query Failed from '{source_frame}' to '{self.world_frame}': {e}", throttle_duration_sec=2.0)
                    continue
            
            # Double-check post-transform values for NaNs
            if any(math.isnan(val) or math.isinf(val) for val in (wx, wy, wz)):
                continue
                
            matched_id = None
            min_dist = float('inf')
            
            for gid, data in self.global_object_db.items():
                if data['class'] == class_name:
                    dist = math.sqrt((wx - data['x'])**2 + (wy - data['y'])**2 + (wz - data['z'])**2)
                    if dist < self.spatial_merge_threshold and dist < min_dist:
                        min_dist = dist
                        matched_id = gid
            
            if matched_id is not None:
                old_data = self.global_object_db[matched_id]
                hits = old_data['hits'] + 1
                self.global_object_db[matched_id]['x'] = (old_data['x'] * old_data['hits'] + wx) / hits
                self.global_object_db[matched_id]['y'] = (old_data['y'] * old_data['hits'] + wy) / hits
                self.global_object_db[matched_id]['z'] = (old_data['z'] * old_data['hits'] + wz) / hits
                self.global_object_db[matched_id]['hits'] = hits
                self.global_object_db[matched_id]['last_seen'] = current_time
                db_updated = True
            else:
                self.global_object_db[self.next_global_id] = {
                    'class': class_name,
                    'x': wx,
                    'y': wy,
                    'z': wz,
                    'hits': 1,
                    'last_seen': current_time
                }
                self.get_logger().info(
                    f"New Global Object [{self.next_global_id}] Registered: '{class_name}' at "
                    f"X:{wx:.2f}, Y:{wy:.2f}, Z:{wz:.2f}"
                )
                self.next_global_id += 1
                db_updated = True
                
        if db_updated:
            self.publish_global_map(current_time)

    def publish_global_map(self, current_time):
        marker_array = MarkerArray()
        for gid, data in self.global_object_db.items():
            sphere = Marker()
            sphere.header.frame_id = self.world_frame
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "global_centroids"
            sphere.id = gid * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=data['x'], y=data['y'], z=data['z'])
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 0.0
            sphere.color.g = 1.0
            sphere.color.b = 0.2
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)
            
            text = Marker()
            text.header.frame_id = self.world_frame
            text.header.stamp = sphere.header.stamp
            text.ns = "global_labels"
            text.id = (gid * 2) + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=data['x'], y=data['y'], z=data['z'] + 0.35)
            text.scale.z = 0.2
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Global ID: G-{gid} | {data['class']} ({data['hits']} hits)"
            marker_array.markers.append(text)
            
        self.global_map_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()