/************************************************************
 *
 * Copyright (c) 2021, University of California, Los Angeles
 *
 * Authors: Kenny J. Chen, Brett T. Lopez
 * Contact: kennyjchen@ucla.edu, btlopez@ucla.edu
 *
 ***********************************************************/

// Project includes
#include <dlo/dlo.h>
#include <frames/frames.h>
#include <frames/map_io.h>

// Internal message types includes
#include <er_nav_msgs/GetCurrentWaypoint.h>
#include <er_nav_msgs/GetWaypointById.h>
#include <er_nav_msgs/SetLocalizationState.h>
#include <er_file_io_msgs/HandleFile.h>

// ROS message type includes
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseArray.h>

// ROS standard interface includes
#include <ros/ros.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <ros/timer.h>
#include <ros/service.h>
#include <ros/service_client.h>

// GTSAM includes
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>

// ROS library includes
#include <tf/transform_listener.h>

// PCL library includes
#include <pcl/common/transforms.h>

// TF libary includes
#include <tf2_ros/static_transform_broadcaster.h>

// c++ standard library includes
#include <memory>

namespace dlo
{
template <typename T>
class OptionalValue
{
public:
    OptionalValue()
    {
        this->value = nullptr;
    }
    OptionalValue(T val)
    {
        this->value = std::make_unique<T>(val);
    }
    operator bool() const
    {
        if (this->value)
        {
            return true;
        }

        return false;
    }
    T operator()() const
    {
        if (this->value)
        {
            return *this->value;
        }

        return T();
    }

private:
    std::unique_ptr<T> value;
};

class OdomNode
{
public:
    OdomNode(ros::NodeHandle node_handle);
    ~OdomNode();

    static void abort()
    {
        abort_ = true;
    }

    void start();
    void stop();

    std::string mapName() const
    {
        return this->map_name;
    }

    unsigned int mapSize() const
    {
        if (!this->keyframes)
        {
            return 0;
        }

        return this->keyframes->size();
    }

    bool loadMap(const std::string& mapname);

private:
    void init();

    void startSubscribers();
    void stopSubscribers();

    void abortTimerCB(const ros::TimerEvent& e);
    void mapPublishTimerCB(const ros::TimerEvent& e);
    void icpCB(const sensor_msgs::PointCloud2ConstPtr& pc);
    void imuCB(const sensor_msgs::Imu::ConstPtr& imu);
    void odomCB(const nav_msgs::Odometry::ConstPtr& odom);

    void setPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose);

    bool saveCallback(er_file_io_msgs::HandleFile::Request& request, er_file_io_msgs::HandleFile::Response& response);
    bool loadCallback(er_file_io_msgs::HandleFile::Request& request, er_file_io_msgs::HandleFile::Response& response);
    bool resetCallback(er_nav_msgs::SetLocalizationState::Request& request, er_nav_msgs::SetLocalizationState::Response& response);
    bool getCurrentWaypointCallback(er_nav_msgs::GetCurrentWaypoint::Request& request, er_nav_msgs::GetCurrentWaypoint::Response& response);
    bool getWaypointByIdCallback(er_nav_msgs::GetWaypointById::Request& request, er_nav_msgs::GetWaypointById::Response& response);

    void getParams();

    template <typename Type>
    bool paramLoadHelper(std::string ns, Type& param, const Type& def_param);
    template <typename Type>
    bool paramLoadHelper(const char* ns, Type& param, const Type& def_param);
    template <typename Type, size_t Length>
    bool vectorParamLoadHelper(std::string ns, std::vector<Type>& param, const std::vector<Type>& def_param);
    template <typename Type, size_t Length>
    bool vectorParamLoadHelper(const char* ns, std::vector<Type>& param, const std::vector<Type>& def_param);

    void publishToROS();
    void publishPose();
    void publishTransform();
    void publishKeyframe();
    void publishAllKeyframes();
    void publishMap();

    void preprocessPoints();
    void initializeInputTarget();
    void setInputSources();

    void initializeDLO();
    void gravityAlign();

    void getNextPose();
    OptionalValue<Eigen::Isometry3f> validatePose(const Eigen::Isometry3f& pose_guess, const pcl::PointCloud<PointType>& scan);
    void integrateIMU();
    void integrateOdom();

    void propagateS2S(Eigen::Matrix4f T);
    void propagateS2M();

    void setAdaptiveParams();

    void computeMetrics();
    void computeSpaciousness();

    void updateKeyframes();

    void optimizeTrajectory();

    double first_imu_time;

    ros::NodeHandle nh;
    ros::Timer abort_timer;
    ros::Timer map_publish_timer;

    ros::Subscriber icp_sub;
    ros::Subscriber imu_sub;
    ros::Subscriber odom_sub;
    ros::Subscriber set_pose_sub;

    ros::Publisher lidar_odom_pub;
    ros::Publisher pose_pub;
    ros::Publisher keyframe_pub;
    ros::Publisher kf_array_pub;
    ros::Publisher kf_pub;
    ros::Publisher full_keyframe_pub;
    ros::Publisher submap_pub;
    ros::Publisher map_pub;

    ros::ServiceServer save_map_service;
    ros::ServiceServer load_map_service;
    ros::ServiceServer get_current_waypoint_service;
    ros::ServiceServer get_waypoint_by_id_service;

    Frames::SubmapParams submap_params;
    std::shared_ptr<Frames> keyframes;

    std::vector<Eigen::Isometry3f> trajectory;

    int gtsam_keyframe_id;
    std::map<int, int> gtsam_kf_id_map;
    gtsam::ISAM2 isam_solver;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_guess;
    gtsam::Values optimized_estimate;

    std::atomic<bool> dlo_initialized;
    std::atomic<bool> imu_calibrated;

    std::string map_directory;
    std::string map_name;
    std::string map_prefix;

    std::string map_frame;
    std::string odom_frame;
    std::string baselink_frame;
    std::string lidar_frame;

    bool publish_keyframe_tf;

    pcl::PointCloud<PointType>::Ptr original_scan;
    pcl::PointCloud<PointType>::Ptr current_scan;

    pcl::PointCloud<PointType>::Ptr keyframe_cloud;
    int num_keyframes;

    pcl::PointCloud<PointType>::Ptr source_cloud;
    pcl::PointCloud<PointType>::Ptr target_cloud;

    ros::Time scan_stamp;

    double curr_frame_stamp;
    double prev_frame_stamp;
    std::vector<double> comp_times;

    nano_gicp::NanoGICP<PointType, PointType> gicp_init;
    nano_gicp::NanoGICP<PointType, PointType> gicp_s2s;
    nano_gicp::NanoGICP<PointType, PointType> gicp;

    pcl::CropBox<PointType> crop;
    pcl::VoxelGrid<PointType> vf_scan;
    pcl::VoxelGrid<PointType> vf_submap;

    Eigen::Matrix4f T;
    Eigen::Matrix4f T_s2s, T_s2s_prev;
    Eigen::Quaternionf q_final;

    Eigen::Vector3f pose_s2s;
    Eigen::Matrix3f rotSO3_s2s;
    Eigen::Quaternionf rotq_s2s;

    Eigen::Vector3f pose;
    Eigen::Matrix3f rotSO3;
    Eigen::Quaternionf rotq;

    Eigen::Matrix4f imu_SE3;
    Eigen::Matrix4f odom_SE3;

    struct XYZd
    {
        double x;
        double y;
        double z;
    };

    struct ImuBias
    {
        XYZd gyro;
        XYZd accel;
    };

    ImuBias imu_bias;

    struct Meas
    {
        double stamp;
    };

    struct ImuMeas : public Meas
    {
        XYZd ang_vel;
        XYZd lin_accel;
    };

    struct OdomMeas : public Meas
    {
        XYZd lin_vel;
        XYZd ang_vel;
    };

    ImuMeas imu_meas;
    OdomMeas odom_meas;

    boost::circular_buffer<ImuMeas> imu_buffer;
    boost::circular_buffer<OdomMeas> odom_buffer;

    static bool comparatorMeas(Meas m1, Meas m2)
    {
        return (m1.stamp < m2.stamp);
    };

    struct Metrics
    {
        std::vector<float> spaciousness;
    };

    Metrics metrics;

    static std::atomic<bool> abort_;

    std::mutex mtx_imu;
    std::mutex mtx_odom;

    // Parameters
    std::string version_;

    bool gravity_align_;

    double keyframe_thresh_dist_;
    double keyframe_thresh_rot_;

    Eigen::Vector3f initial_position_;
    Eigen::Quaternionf initial_orientation_;

    Eigen::Isometry3d baselink_tf_lidar_;

    bool crop_use_;
    double crop_size_;

    bool vf_scan_use_;
    double vf_scan_res_;

    bool vf_submap_use_;
    double vf_submap_res_;

    bool adaptive_params_use_;

    bool odom_use_;
    int odom_buffer_size_;

    bool imu_use_;
    int imu_calib_time_;
    int imu_buffer_size_;

    struct InitializationParams
    {
        float max_distance_threshold_m_;
        float max_angle_threshold_rad_;
        float max_icp_fitness_score_;
    };

    InitializationParams init_params;

    bool trajectory_optimization_use_;

    double map_publish_freq_;
    double map_vf_leaf_size_;

    int gicp_min_num_points_;

    int gicps2s_k_correspondences_;
    double gicps2s_max_corr_dist_;
    int gicps2s_max_iter_;
    double gicps2s_transformation_ep_;
    double gicps2s_euclidean_fitness_ep_;
    int gicps2s_ransac_iter_;
    double gicps2s_ransac_inlier_thresh_;

    int gicps2m_k_correspondences_;
    double gicps2m_max_corr_dist_;
    int gicps2m_max_iter_;
    double gicps2m_transformation_ep_;
    double gicps2m_euclidean_fitness_ep_;
    int gicps2m_ransac_iter_;
    double gicps2m_ransac_inlier_thresh_;
};

static geometry_msgs::Pose toRosMsg(const Eigen::Isometry3f& eigen_pose)
{
    geometry_msgs::Pose pose_msg;
    Eigen::Quaternionf quat(eigen_pose.rotation());

    pose_msg.position.x = eigen_pose.translation().x();
    pose_msg.position.y = eigen_pose.translation().y();
    pose_msg.position.z = eigen_pose.translation().z();
    pose_msg.orientation.w = quat.w();
    pose_msg.orientation.x = quat.x();
    pose_msg.orientation.y = quat.y();
    pose_msg.orientation.z = quat.z();

    return pose_msg;
}
}  // namespace dlo
