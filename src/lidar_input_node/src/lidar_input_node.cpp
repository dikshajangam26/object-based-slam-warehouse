#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <laser_geometry/laser_geometry.hpp>

class LidarInputNode : public rclcpp::Node {
public:
    LidarInputNode() : Node("lidar_input_node") {
        // Subscribe to the simulation's raw laser output
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/cpr_r100_0000/sensors/lidar_0/scan", rclcpp::SensorDataQoS(),
            std::bind(&LidarInputNode::scan_callback, this, std::placeholders::_1));
        
            // Publish formatted PointCloud2
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/pointcloud", 10);
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
        sensor_msgs::msg::PointCloud2 pc_msg;
        projector_.projectLaser(*scan_msg, pc_msg);

        // Explicitly set the frame to 'lidar_link'
        pc_msg.header.frame_id = "lidar_link";
        pc_msg.header.stamp = scan_msg->header.stamp;

        pc_pub_->publish(pc_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    laser_geometry::LaserProjection projector_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarInputNode>());
    rclcpp::shutdown();
    return 0;
}