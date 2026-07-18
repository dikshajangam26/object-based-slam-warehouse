#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>

// Time synchronization headers
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/pass_through.h>
#include <rmw/qos_profiles.h>

// Core ORB-SLAM3 library header
#include <System.h>

class OrbSlam3Node : public rclcpp::Node {
public:
    OrbSlam3Node() : Node("orb_slam3_node") {
        this->declare_parameter<std::string>("voc_file", "");
        this->declare_parameter<std::string>("settings_file", "");

        std::string voc_file = this->get_parameter("voc_file").as_string();
        std::string settings_file = this->get_parameter("settings_file").as_string();

        if (voc_file.empty() || settings_file.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Vocabulary or settings file path is empty!");
            rclcpp::shutdown();
        }

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);

        RCLCPP_INFO(this->get_logger(), "Loading ORB-SLAM3 Engine (RGB-D)...");
        slam_engine_ = new ORB_SLAM3::System(voc_file, settings_file, ORB_SLAM3::System::RGBD, true);
        RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 Engine ready.");

        // Link the Synchronizer to our PassThrough middle-men
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), rgb_pass_, depth_pass_);
        sync_->registerCallback(std::bind(&OrbSlam3Node::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();

        // BYPASS: Use std::bind instead of lambdas
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/rgb/image", sensor_qos,
            std::bind(&OrbSlam3Node::rgb_callback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/depth/image", sensor_qos,
            std::bind(&OrbSlam3Node::depth_callback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Subscriptions active. Awaiting synchronized frames...");
    }

    ~OrbSlam3Node() {
        slam_engine_->Shutdown();
        delete slam_engine_;
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;

    // Standard member functions to avoid compiler lambda crashes
    void rgb_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        rgb_pass_.add(msg);
    }

    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        depth_pass_.add(msg);
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        
        cv_bridge::CvImagePtr cv_ptr_rgb;
        cv_bridge::CvImagePtr cv_ptr_depth;

        try {
            cv_ptr_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
            cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        double timestamp = rclcpp::Time(rgb_msg->header.stamp).seconds();

        // Pass OpenCV matrices to ORB-SLAM3
        Sophus::SE3f Tcw = slam_engine_->TrackRGBD(cv_ptr_rgb->image, cv_ptr_depth->image, timestamp);
        
        int state = slam_engine_->GetTrackingState();
        
        if (state == 2) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = rgb_msg->header.stamp;
            pose_msg.header.frame_id = "map";

            Eigen::Matrix4f Tww = Tcw.inverse().matrix();
            pose_msg.pose.position.x = Tww(0,3);
            pose_msg.pose.position.y = Tww(1,3);
            pose_msg.pose.position.z = Tww(2,3);

            Eigen::Quaternionf q(Tww.block<3,3>(0,0));
            pose_msg.pose.orientation.x = q.x();
            pose_msg.pose.orientation.y = q.y();
            pose_msg.pose.orientation.z = q.z();
            pose_msg.pose.orientation.w = q.w();

            pose_pub_->publish(pose_msg);
        }
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    message_filters::PassThrough<sensor_msgs::msg::Image> rgb_pass_;
    message_filters::PassThrough<sensor_msgs::msg::Image> depth_pass_;
    
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    ORB_SLAM3::System* slam_engine_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OrbSlam3Node>());
    rclcpp::shutdown();
    return 0;
}