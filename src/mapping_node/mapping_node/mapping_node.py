import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import math
import json
import os
import csv

# TF2 Libraries
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        
        # Database Format: { global_id: {'class', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'volume', 'is_on_ground', 'hits', 'last_seen'} }
        self.global_object_db = {}
        self.next_global_id = 1
        self.spatial_merge_threshold = 0.4  # 40cm Euclidean distance for deduplication
        self.ground_level_threshold = 0.30  # Bottom edge <= 30cm tagged as grounded
        
        # Task 4.5 Persistence Configuration (JSON & CSV Paths)
        self.map_file_path = os.path.expanduser('~/slam_ws/semantic_map.json')
        self.csv_file_path = os.path.expanduser('~/slam_ws/src/data_logging/warehouse_inventory.csv')
        
        # Explicit 3-Box Color Palette (RGB)
        self.metadata_colors = {
            'green_box':  (0.0, 0.85, 0.2),  # Vibrant Green
            'green':      (0.0, 0.85, 0.2),
            'yellow_box': (0.95, 0.85, 0.1), # Warning Yellow
            'yellow':     (0.95, 0.85, 0.1),
            'orange_box': (1.0, 0.55, 0.0),  # Industrial Orange
            'orange':     (1.0, 0.55, 0.0),
            'default':    (0.7, 0.7, 0.7)    # Neutral Grey fallback
        }
        
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
        
        # Load stored map data if it exists from a previous session
        self.load_map_from_disk()
        
        # 1 Hz Timer to continuously broadcast the map
        self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("Semantic Mapping Node Online. 3-Box Classification, JSON & CSV Logging Active.")

    def timer_callback(self):
        if self.global_object_db:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            self.publish_global_map(current_time)

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
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            dx, dy, dz = 0.5, 0.5, 0.5
        volume = dx * dy * dz
        z_bottom = z_centroid - (dz / 2.0)
        is_on_ground = bool(z_bottom <= self.ground_level_threshold and z_centroid > -0.5)
        return dx, dy, dz, volume, is_on_ground

    def resolve_class_name(self, marker, z_centroid, is_on_ground):
        raw_name = marker.text.strip().lower() if marker.text else ""
        if not raw_name and marker.ns and marker.ns.lower() != "default":
            raw_name = marker.ns.strip().lower()
            
        for valid_key in ['green', 'yellow', 'orange']:
            if valid_key in raw_name:
                return f"{valid_key}_box"
                
        if is_on_ground:
            return 'green_box'
        elif z_centroid <= 1.0:
            return 'yellow_box'
        else:
            return 'orange_box'

    def centroid_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        db_updated = False
        
        for marker in msg.markers:
            source_frame = marker.header.frame_id
            lx, ly, lz = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
            
            if any(math.isnan(val) or math.isinf(val) for val in (lx, ly, lz)):
                continue
                
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
                
            raw_dx, raw_dy, raw_dz = marker.scale.x, marker.scale.y, marker.scale.z
            dx, dy, dz, volume, is_on_ground = self.infer_semantic_properties(wz, raw_dx, raw_dy, raw_dz)
            class_name = self.resolve_class_name(marker, wz, is_on_ground)
                
            matched_id = None
            min_dist = float('inf')
            
            for gid, data in self.global_object_db.items():
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
                _, _, _, vol, grounded = self.infer_semantic_properties(new_z, old_data['dx'], old_data['dy'], old_data['dz'])
                
                updated_class = class_name if old_data['class'] == 'default' else old_data['class']
                
                self.global_object_db[matched_id].update({
                    'class': updated_class, 'x': new_x, 'y': new_y, 'z': new_z,
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
                    f"New Structural Element [{self.next_global_id}] '{class_name.upper()}' | "
                    f"Vol: {volume:.2f}m³ | Grounded: {is_on_ground}"
                )
                self.next_global_id += 1
                db_updated = True
                
        if db_updated:
            self.publish_global_map(current_time)

    def publish_global_map(self, current_time):
        marker_array = MarkerArray()
        for gid, data in self.global_object_db.items():
            color_rgb = self.metadata_colors.get(data['class'], self.metadata_colors['default'])
            
            bbox = Marker()
            bbox.header.frame_id = self.world_frame
            bbox.header.stamp = self.get_clock().now().to_msg()
            bbox.ns = f"category_{data['class']}"
            bbox.id = gid * 2
            bbox.type = Marker.CUBE
            bbox.action = Marker.ADD
            bbox.pose.position = Point(x=data['x'], y=data['y'], z=data['z'])
            bbox.scale.x = data['dx']
            bbox.scale.y = data['dy']
            bbox.scale.z = data['dz']
            bbox.color.r, bbox.color.g, bbox.color.b = color_rgb
            bbox.color.a = 0.55 if data['is_on_ground'] else 0.85
            marker_array.markers.append(bbox)
            
            text = Marker()
            text.header.frame_id = self.world_frame
            text.header.stamp = bbox.header.stamp
            text.ns = "metadata_labels"
            text.id = (gid * 2) + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=data['x'], y=data['y'], z=data['z'] + (data['dz'] / 2.0) + 0.25)
            text.scale.z = 0.22
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            text.text = (
                f"[{data['class'].upper()}] G-{gid} ({data['hits']} hits)\n"
                f"Vol: {data['volume']:.2f}m³ | Grounded: {data['is_on_ground']}"
            )
            marker_array.markers.append(text)
            
        self.global_map_pub.publish(marker_array)

    def load_map_from_disk(self):
        if not os.path.exists(self.map_file_path):
            self.get_logger().info("No prior map file found. Starting fresh map session.")
            return

        try:
            with open(self.map_file_path, 'r') as f:
                saved_data = json.load(f)
            self.global_object_db = {int(k): v for k, v in saved_data.items()}
            if self.global_object_db:
                self.next_global_id = max(self.global_object_db.keys()) + 1
                self.get_logger().info(f"Successfully loaded {len(self.global_object_db)} objects from {self.map_file_path}")
                self.publish_global_map(0.0)
        except Exception as e:
            self.get_logger().error(f"Failed to read map file {self.map_file_path}: {e}")

    def save_map_to_disk(self):
        """Save structured JSON dictionary to disk."""
        try:
            os.makedirs(os.path.dirname(self.map_file_path), exist_ok=True)
            with open(self.map_file_path, 'w') as f:
                json.dump(self.global_object_db, f, indent=4)
            self.get_logger().info(f"Cleanly flushed {len(self.global_object_db)} mapped objects to JSON: {self.map_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON map data to disk: {e}")

    def save_csv_to_disk(self):
        """Export engineering inventory dataset directly to CSV in data_logging folder."""
        try:
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            with open(self.csv_file_path, mode='w', newline='') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    'Global_ID', 'Class_Name', 'X_m', 'Y_m', 'Z_m', 
                    'Volume_m3', 'Structural_Status', 'Observation_Hits', 'Last_Seen_Timestamp'
                ])
                for gid, item in sorted(self.global_object_db.items(), key=lambda x: int(x[0])):
                    status = "Floor" if item['is_on_ground'] else "Elevated"
                    writer.writerow([
                        f"G-{gid}",
                        item['class'],
                        round(item['x'], 4),
                        round(item['y'], 4),
                        round(item['z'], 4),
                        round(item['volume'], 6),
                        status,
                        item['hits'],
                        round(item['last_seen'], 2)
                    ])
            self.get_logger().info(f"Cleanly flushed CSV dataset to: {self.csv_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV dataset to disk: {e}")

    def destroy_node(self):
        # Guarantee both JSON and CSV files trigger simultaneously on shutdown
        self.save_map_to_disk()
        self.save_csv_to_disk()
        super().destroy_node()

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