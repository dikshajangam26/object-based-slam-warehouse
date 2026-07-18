#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class ZedInputNode : public rclcpp::Node {
public:
    ZedInputNode() : Node("zed_input_node") {
        // Create a Best Effort QoS profile designed specifically for heavy sensor data
        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();

        // Output Publishers (Now using sensor_qos instead of '10')
        rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/zed/rgb/image", sensor_qos);
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/zed/depth/image", sensor_qos);
        info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/zed/camera_info", sensor_qos);

        // Input Subscribers (Now using sensor_qos instead of '10')
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "sim_rgb_in", sensor_qos, std::bind(&ZedInputNode::rgb_cb, this, std::placeholders::_1));
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "sim_depth_in", sensor_qos, std::bind(&ZedInputNode::depth_cb, this, std::placeholders::_1));
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "sim_info_in", sensor_qos, std::bind(&ZedInputNode::info_cb, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "ZED Input Node running. SensorDataQoS active.");
    }

private:
    void rgb_cb(const sensor_msgs::msg::Image::SharedPtr msg) { rgb_pub_->publish(*msg); }
    void depth_cb(const sensor_msgs::msg::Image::SharedPtr msg) { depth_pub_->publish(*msg); }
    void info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) { info_pub_->publish(*msg); }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_, depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_, depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ZedInputNode>());
    rclcpp::shutdown();
    return 0;
}