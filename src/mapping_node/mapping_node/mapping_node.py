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
        
        # Database Format: { global_id: {'class', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'volume', 'is_on_ground', 'hits', 'last_seen'} }
        self.global_object_db = {}
        self.next_global_id = 1
        self.spatial_merge_threshold = 0.4 # 40cm Euclidean distance for deduplication
        self.ground_level_threshold = 0.30 # Objects with bottom edge <= 30cm from Z=0 are tagged as grounded
        
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
        self.get_logger().info("Semantic Mapping Node Online. Bounding & Ground Inference Active.")

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

    def infer_semantic_properties(self, z_centroid, dx, dy, dz):
        """Task 4.2: Calculate volumetric profiles and structural ground positioning."""
        # Fallback if upstream reconstruction node sends 0 dimensions
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            dx, dy, dz = 0.5, 0.5, 0.5
            
        volume = dx * dy * dz
        
        # Calculate where the bottom edge of the box touches the world
        z_bottom = z_centroid - (dz / 2.0)
        
        # Check if resting on Gazebo structural grid layer (approx Z = 0.0 in odom frame)
        is_on_ground = bool(z_bottom <= self.ground_level_threshold and z_centroid > -0.5)
        
        return dx, dy, dz, volume, is_on_ground

    def centroid_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        db_updated = False
        
        for marker in msg.markers:
            source_frame = marker.header.frame_id
            lx, ly, lz = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
            
            # Filter out NaN or Infinite coordinates from reflective/empty depth pixels
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
            
            if any(math.isnan(val) or math.isinf(val) for val in (wx, wy, wz)):
                continue
                
            # Extract bounding dimensions from marker scale
            raw_dx, raw_dy, raw_dz = marker.scale.x, marker.scale.y, marker.scale.z
            dx, dy, dz, volume, is_on_ground = self.infer_semantic_properties(wz, raw_dx, raw_dy, raw_dz)
                
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
                new_x = (old_data['x'] * old_data['hits'] + wx) / hits
                new_y = (old_data['y'] * old_data['hits'] + wy) / hits
                new_z = (old_data['z'] * old_data['hits'] + wz) / hits
                
                # Re-evaluate semantic properties with smoothed Z height
                _, _, _, vol, grounded = self.infer_semantic_properties(new_z, old_data['dx'], old_data['dy'], old_data['dz'])
                
                self.global_object_db[matched_id].update({
                    'x': new_x, 'y': new_y, 'z': new_z,
                    'volume': vol, 'is_on_ground': grounded,
                    'hits': hits, 'last_seen': current_time
                })
                db_updated = True
            else:
                self.global_object_db[self.next_global_id] = {
                    'class': class_name,
                    'x': wx, 'y': wy, 'z': wz,
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'volume': volume,
                    'is_on_ground': is_on_ground,
                    'hits': 1,
                    'last_seen': current_time
                }
                self.get_logger().info(
                    f"New Structural Element [{self.next_global_id}] '{class_name}' | "
                    f"Vol: {volume:.2f}m³ | Grounded: {is_on_ground}"
                )
                self.next_global_id += 1
                db_updated = True
                
        if db_updated:
            self.publish_global_map(current_time)

    def publish_global_map(self, current_time):
        marker_array = MarkerArray()
        for gid, data in self.global_object_db.items():
            # Draw a 3D Bounding Box (CUBE) instead of a sphere to visualize structural volume
            bbox = Marker()
            bbox.header.frame_id = self.world_frame
            bbox.header.stamp = self.get_clock().now().to_msg()
            bbox.ns = "global_bounding_boxes"
            bbox.id = gid * 2
            bbox.type = Marker.CUBE
            bbox.action = Marker.ADD
            bbox.pose.position = Point(x=data['x'], y=data['y'], z=data['z'])
            bbox.scale.x = data['dx']
            bbox.scale.y = data['dy']
            bbox.scale.z = data['dz']
            
            # Color coding: Green if resting on the ground grid, Orange if elevated (e.g., shelf)
            if data['is_on_ground']:
                bbox.color.r = 0.0; bbox.color.g = 0.85; bbox.color.b = 0.2
            else:
                bbox.color.r = 1.0; bbox.color.g = 0.55; bbox.color.b = 0.0
            bbox.color.a = 0.65
            marker_array.markers.append(bbox)
            
            # Hovering Semantic Tag
            text = Marker()
            text.header.frame_id = self.world_frame
            text.header.stamp = bbox.header.stamp
            text.ns = "global_labels"
            text.id = (gid * 2) + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=data['x'], y=data['y'], z=data['z'] + (data['dz'] / 2.0) + 0.25)
            text.scale.z = 0.2
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            text.text = (
                f"Global ID: G-{gid} | {data['class']} ({data['hits']} hits)\n"
                f"Vol: {data['volume']:.2f}m³ | Grounded: {data['is_on_ground']}"
            )
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