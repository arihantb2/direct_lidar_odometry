/************************************************************
 *
 * Copyright (c) 2021, University of California, Los Angeles
 *
 * Authors: Kenny J. Chen, Brett T. Lopez
 * Contact: kennyjchen@ucla.edu, btlopez@ucla.edu
 *
 ***********************************************************/

#include <dlo/odom.h>

namespace dlo
{
std::atomic<bool> OdomNode::abort_(false);

/**
 * Constructor
 **/

OdomNode::OdomNode(ros::NodeHandle node_handle) : nh(node_handle)
{
    this->getParams();

    // Initialize ROS publishers
    this->lidar_odom_pub = this->nh.advertise<nav_msgs::Odometry>("odom", 1);
    this->pose_pub = this->nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
    this->kf_pub = this->nh.advertise<nav_msgs::Odometry>("keyframe_odom", 1, true);
    this->kf_array_pub = this->nh.advertise<geometry_msgs::PoseArray>("keyframe_array", 1, true);
    this->keyframe_pub = this->nh.advertise<sensor_msgs::PointCloud2>("keyframe", 1, true);
    this->submap_pub = this->nh.advertise<sensor_msgs::PointCloud2>("submap", 1, true);
    this->map_pub = this->nh.advertise<sensor_msgs::PointCloud2>("map", 1, true);

    // Initialize set pose subscriber
    this->set_pose_sub =
        this->nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("set_pose", 1U, &OdomNode::setPoseCB, this, ros::TransportHints().tcpNoDelay());

    // Initialize ROS services
    this->reset_service = this->nh.advertiseService("/localization/reset", &OdomNode::resetCallback, this);
    this->save_map_service = this->nh.advertiseService("/localization/save_map", &OdomNode::saveCallback, this);
    // this->load_map_service = this->nh.advertiseService("/localization/load_map", &OdomNode::loadCallback, this);
    // this->get_current_waypoint_service = this->nh.advertiseService("/localization/get_current_waypoint", &OdomNode::getCurrentWaypointCallback, this);
    // this->get_waypoint_by_id_service = this->nh.advertiseService("/localization/get_waypoint_by_id", &OdomNode::getWaypointByIdCallback, this);

    this->imu_calibrated = false;

    this->init();

    this->loadMap(this->map_name);

    this->startSubscribers();
}

/**
 * Destructor
 **/

OdomNode::~OdomNode()
{
}

/**
 * Odom Node Parameters
 **/

void OdomNode::getParams()
{
    // Version
    ros::param::param<std::string>("~dlo/version", this->version_, "0.0.0");

    // Frames
    ros::param::param<std::string>("~dlo/odomNode/map_frame", this->map_frame, "map");
    ros::param::param<std::string>("~dlo/odomNode/odom_frame", this->odom_frame, "odom");
    ros::param::param<std::string>("~dlo/odomNode/baselink_frame", this->baselink_frame, "base_link");
    ros::param::param<std::string>("~dlo/odomNode/lidar_frame", this->lidar_frame, "lidar_link");

    // Map IO directory
    ros::param::param<std::string>("~dlo/mapDirectory", this->map_directory, "");
    ros::param::param<std::string>("~dlo/mapName", this->map_name, "");

    // Gravity alignment
    ros::param::param<bool>("~dlo/gravityAlign", this->gravity_align_, false);

    // Keyframe Threshold
    ros::param::param<double>("~dlo/odomNode/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
    ros::param::param<double>("~dlo/odomNode/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

    // Submap
    ros::param::param<int>("~dlo/odomNode/submap/keyframe/knn", this->submap_params.submap_knn_, 10);
    ros::param::param<int>("~dlo/odomNode/submap/keyframe/kcv", this->submap_params.submap_kcv_, 10);
    ros::param::param<int>("~dlo/odomNode/submap/keyframe/kcc", this->submap_params.submap_kcc_, 10);

    // Initial Pose
    double px, py, pz, qx, qy, qz, qw;
    ros::param::param<double>("~dlo/odomNode/initialPose/position/x", px, 0.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/position/y", py, 0.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/position/z", pz, 0.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/orientation/w", qw, 1.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/orientation/x", qx, 0.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/orientation/y", qy, 0.0);
    ros::param::param<double>("~dlo/odomNode/initialPose/orientation/z", qz, 0.0);
    this->initial_position_ = Eigen::Vector3f(px, py, pz);
    this->initial_orientation_ = Eigen::Quaternionf(qw, qx, qy, qz);

    // Initialization validation thresholds
    ros::param::param<float>("~dlo/odomNode/max_initialization_distance_threshold_m", this->max_initialization_distance_threshold_m_, 5.0);
    ros::param::param<float>("~dlo/odomNode/max_initialization_angle_threshold_rad", this->max_initialization_angle_threshold_rad_, 10.0 * M_PI / 180.0);

    // Crop Box Filter
    ros::param::param<bool>("~dlo/odomNode/preprocessing/cropBoxFilter/use", this->crop_use_, false);
    ros::param::param<double>("~dlo/odomNode/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

    // Voxel Grid Filter
    ros::param::param<bool>("~dlo/odomNode/preprocessing/voxelFilter/scan/use", this->vf_scan_use_, true);
    ros::param::param<double>("~dlo/odomNode/preprocessing/voxelFilter/scan/res", this->vf_scan_res_, 0.05);
    ros::param::param<bool>("~dlo/odomNode/preprocessing/voxelFilter/submap/use", this->vf_submap_use_, false);
    ros::param::param<double>("~dlo/odomNode/preprocessing/voxelFilter/submap/res", this->vf_submap_res_, 0.1);

    // Adaptive Parameters
    ros::param::param<bool>("~dlo/adaptiveParams", this->adaptive_params_use_, false);

    // Odom
    ros::param::param<bool>("~dlo/odom", this->odom_use_, false);
    ros::param::param<int>("~dlo/odomNode/odom/bufferSize", this->odom_buffer_size_, 2000);

    // IMU
    ros::param::param<bool>("~dlo/imu", this->imu_use_, false);
    ros::param::param<int>("~dlo/odomNode/imu/calibTime", this->imu_calib_time_, 3);
    ros::param::param<int>("~dlo/odomNode/imu/bufferSize", this->imu_buffer_size_, 2000);

    // Trajectory Optimization
    ros::param::param<bool>("~dlo/trajectoryOptimization", this->trajectory_optimization_use_, false);

    // Map
    ros::param::param<double>("~dlo/mapNode/publishFreq", this->map_publish_freq_, 1.0);
    ros::param::param<double>("~dlo/mapNode/leafSize", this->map_vf_leaf_size_, 0.5);
    ros::param::param<std::string>("~dlo/mapNode/mapPrefix", this->map_prefix, "exr2");

    // GICP
    ros::param::param<int>("~dlo/odomNode/gicp/minNumPoints", this->gicp_min_num_points_, 100);
    ros::param::param<int>("~dlo/odomNode/gicp/s2s/kCorrespondences", this->gicps2s_k_correspondences_, 20);
    ros::param::param<double>("~dlo/odomNode/gicp/s2s/maxCorrespondenceDistance", this->gicps2s_max_corr_dist_, std::sqrt(std::numeric_limits<double>::max()));
    ros::param::param<int>("~dlo/odomNode/gicp/s2s/maxIterations", this->gicps2s_max_iter_, 64);
    ros::param::param<double>("~dlo/odomNode/gicp/s2s/transformationEpsilon", this->gicps2s_transformation_ep_, 0.0005);
    ros::param::param<double>("~dlo/odomNode/gicp/s2s/euclideanFitnessEpsilon", this->gicps2s_euclidean_fitness_ep_, -std::numeric_limits<double>::max());
    ros::param::param<int>("~dlo/odomNode/gicp/s2s/ransac/iterations", this->gicps2s_ransac_iter_, 0);
    ros::param::param<double>("~dlo/odomNode/gicp/s2s/ransac/outlierRejectionThresh", this->gicps2s_ransac_inlier_thresh_, 0.05);
    ros::param::param<int>("~dlo/odomNode/gicp/s2m/kCorrespondences", this->gicps2m_k_correspondences_, 20);
    ros::param::param<double>("~dlo/odomNode/gicp/s2m/maxCorrespondenceDistance", this->gicps2m_max_corr_dist_, std::sqrt(std::numeric_limits<double>::max()));
    ros::param::param<int>("~dlo/odomNode/gicp/s2m/maxIterations", this->gicps2m_max_iter_, 64);
    ros::param::param<double>("~dlo/odomNode/gicp/s2m/transformationEpsilon", this->gicps2m_transformation_ep_, 0.0005);
    ros::param::param<double>("~dlo/odomNode/gicp/s2m/euclideanFitnessEpsilon", this->gicps2m_euclidean_fitness_ep_, -std::numeric_limits<double>::max());
    ros::param::param<int>("~dlo/odomNode/gicp/s2m/ransac/iterations", this->gicps2m_ransac_iter_, 0);
    ros::param::param<double>("~dlo/odomNode/gicp/s2m/ransac/outlierRejectionThresh", this->gicps2m_ransac_inlier_thresh_, 0.05);

    tf::TransformListener tf_listener;
    tf::StampedTransform transform;
    try
    {
        tf_listener.waitForTransform(this->baselink_frame, this->lidar_frame, ros::Time(0), ros::Duration(5.0));
        tf_listener.lookupTransform(this->baselink_frame, this->lidar_frame, ros::Time(0), transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR_STREAM("Cannot get transform [" << this->baselink_frame << " -> " << this->lidar_frame << "]. Error: [" << ex.what() << "]");
    }

    tf::transformTFToEigen(transform, baselink_tf_lidar_);

    std::cout << "[OdomNode::getParams]: [" << this->baselink_frame << "] to [" << this->lidar_frame << "] transform: [position ("
              << baselink_tf_lidar_.translation().transpose() << "), orientation (" << baselink_tf_lidar_.rotation().eulerAngles(0, 1, 2).transpose() << ")]\n";
}

void OdomNode::init()
{
    this->dlo_initialized = false;

    // Initialize class variables
    this->T = Eigen::Matrix4f::Identity();
    this->T_s2s = Eigen::Matrix4f::Identity();
    this->T_s2s_prev = Eigen::Matrix4f::Identity();

    this->pose_s2s = Eigen::Vector3f(0., 0., 0.);
    this->rotSO3_s2s = Eigen::Matrix3f::Identity();
    this->rotq_s2s = Eigen::Quaternionf(1., 0., 0., 0.);

    this->pose = Eigen::Vector3f(0., 0., 0.);
    this->rotSO3 = Eigen::Matrix3f::Identity();
    this->rotq = Eigen::Quaternionf(1., 0., 0., 0.);

    this->gtsam_keyframe_id = 0;

    this->isam_solver = gtsam::ISAM2(gtsam::ISAM2Params(gtsam::ISAM2GaussNewtonParams(), 0.1, 1));
    this->initial_guess.clear();
    this->optimized_estimate.clear();
    this->graph.resize(0);

    this->imu_SE3 = Eigen::Matrix4f::Identity();

    this->imu_bias.gyro.x = 0.;
    this->imu_bias.gyro.y = 0.;
    this->imu_bias.gyro.z = 0.;
    this->imu_bias.accel.x = 0.;
    this->imu_bias.accel.y = 0.;
    this->imu_bias.accel.z = 0.;

    this->imu_meas.stamp = 0.;
    this->imu_meas.ang_vel.x = 0.;
    this->imu_meas.ang_vel.y = 0.;
    this->imu_meas.ang_vel.z = 0.;
    this->imu_meas.lin_accel.x = 0.;
    this->imu_meas.lin_accel.y = 0.;
    this->imu_meas.lin_accel.z = 0.;

    this->imu_buffer.clear();
    this->imu_buffer.set_capacity(this->imu_buffer_size_);
    this->first_imu_time = 0.;

    this->odom_SE3 = Eigen::Matrix4f::Identity();

    this->odom_meas.stamp = 0.;
    this->odom_meas.lin_vel.x = 0.;
    this->odom_meas.lin_vel.y = 0.;
    this->odom_meas.lin_vel.z = 0.;
    this->odom_meas.ang_vel.x = 0.;
    this->odom_meas.ang_vel.y = 0.;
    this->odom_meas.ang_vel.z = 0.;

    this->odom_buffer.clear();
    this->odom_buffer.set_capacity(this->odom_buffer_size_);

    this->original_scan = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    this->current_scan = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);

    this->keyframe_cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);

    this->source_cloud = nullptr;
    this->target_cloud = nullptr;

    this->keyframes = std::make_shared<Frames>(submap_params);
    this->keyframes->setCvHullAlpha(this->keyframe_thresh_dist_);

    this->gicp_s2s.setCorrespondenceRandomness(this->gicps2s_k_correspondences_);
    this->gicp_s2s.setMaxCorrespondenceDistance(this->gicps2s_max_corr_dist_);
    this->gicp_s2s.setMaximumIterations(this->gicps2s_max_iter_);
    this->gicp_s2s.setTransformationEpsilon(this->gicps2s_transformation_ep_);
    this->gicp_s2s.setEuclideanFitnessEpsilon(this->gicps2s_euclidean_fitness_ep_);
    this->gicp_s2s.setRANSACIterations(this->gicps2s_ransac_iter_);
    this->gicp_s2s.setRANSACOutlierRejectionThreshold(this->gicps2s_ransac_inlier_thresh_);

    this->gicp.setCorrespondenceRandomness(this->gicps2m_k_correspondences_);
    this->gicp.setMaxCorrespondenceDistance(this->gicps2m_max_corr_dist_);
    this->gicp.setMaximumIterations(this->gicps2m_max_iter_);
    this->gicp.setTransformationEpsilon(this->gicps2m_transformation_ep_);
    this->gicp.setEuclideanFitnessEpsilon(this->gicps2m_euclidean_fitness_ep_);
    this->gicp.setRANSACIterations(this->gicps2m_ransac_iter_);
    this->gicp.setRANSACOutlierRejectionThreshold(this->gicps2m_ransac_inlier_thresh_);

    pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
    this->gicp_s2s.setSearchMethodSource(temp, true);
    this->gicp_s2s.setSearchMethodTarget(temp, true);
    this->gicp.setSearchMethodSource(temp, true);
    this->gicp.setSearchMethodTarget(temp, true);

    this->crop.setNegative(true);
    this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
    this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

    this->vf_scan.setLeafSize(this->vf_scan_res_, this->vf_scan_res_, this->vf_scan_res_);
    this->vf_submap.setLeafSize(this->vf_submap_res_, this->vf_submap_res_, this->vf_submap_res_);

    this->metrics.spaciousness.clear();
    this->metrics.spaciousness.push_back(0.);

    ROS_INFO("DLO Odom Node Initialized");
}

bool OdomNode::loadMap(const std::string& mapname)
{
    if (mapname.empty())
    {
        return false;
    }

    std::string map_path = this->map_directory + "/" + mapname;
    if (boost::filesystem::exists(map_path) && boost::filesystem::is_directory(map_path))
    {
        KeyframeMap<int> frames = loadMapFramesFromDisk(map_path);
        if (frames.empty())
        {
            std::cout << "[OdomNode::loadMap]: Requested map [" << mapname << "] is empty. Failed to load a usable map\n";
            return false;
        }

        this->map_name = mapname;
        this->keyframes.reset();
        this->keyframes = std::make_shared<Frames>(this->submap_params, frames);
        this->keyframes->setCvHullAlpha(this->keyframe_thresh_dist_);

        this->publishAllKeyframes();

        std::cout << "[OdomNode::loadMap]: Loaded map [" << mapname << "]\n";
        return true;
    }

    std::cout << "[OdomNode::loadMap]: Requested map [" << mapname << "] not found. Failed to load map\n";
    return false;
}

void OdomNode::startSubscribers()
{
    // Initialize ROS subscribers
    this->icp_sub = this->nh.subscribe("pointcloud", 1, &OdomNode::icpCB, this);
    this->imu_sub = this->nh.subscribe("imu", 1, &OdomNode::imuCB, this);
    this->odom_sub = this->nh.subscribe("wheel_odom", 1, &OdomNode::odomCB, this);

    // Intialize ROS timers
    this->map_publish_timer = this->nh.createTimer(1.0 / this->map_publish_freq_, &OdomNode::mapPublishTimerCB, this);
}

void OdomNode::stopSubscribers()
{
    this->icp_sub.shutdown();
    this->imu_sub.shutdown();
    this->odom_sub.shutdown();

    this->map_publish_timer.stop();

    ros::getGlobalCallbackQueue()->clear();
}

/**
 * Start Odom Thread
 **/

void OdomNode::start()
{
    ROS_INFO("Starting DLO Odometry Node");

    // printf("\033[2J\033[1;1H");
    std::cout << std::endl << "==== Direct LiDAR Odometry v" << this->version_ << " ====" << std::endl << std::endl;
}

/**
 * Stop Odom Thread
 **/

void OdomNode::stop()
{
    ROS_WARN("Stopping DLO Odometry Node");
    ros::shutdown();
}

/**
 * Abort Timer Callback
 **/

void OdomNode::abortTimerCB(const ros::TimerEvent& e)
{
    if (abort_)
    {
        stop();
    }
}

/*
 * Map publish timer callback
 */

void OdomNode::mapPublishTimerCB(const ros::TimerEvent& e)
{
    publishMap();
}

void OdomNode::publishMap()
{
    if (this->map_pub.getNumSubscribers() == 0)
    {
        return;
    }

    if (this->keyframes == nullptr)
    {
        return;
    }

    pcl::PointCloud<PointType>::Ptr map_cloud = this->keyframes->getMapDS(this->map_vf_leaf_size_);

    if (map_cloud == nullptr)
    {
        return;
    }

    if (map_cloud->empty())
    {
        return;
    }

    sensor_msgs::PointCloud2 map_ros;
    pcl::toROSMsg(*map_cloud, map_ros);
    map_ros.header.stamp = ros::Time::now();
    map_ros.header.frame_id = this->map_frame;
    this->map_pub.publish(map_ros);
}

/**
 * Publish to ROS
 **/

void OdomNode::publishToROS()
{
    this->publishPose();
    this->publishTransform();

    if (this->keyframes->submapHasChanged() && this->submap_pub.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*this->keyframes->submapCloud(), cloud_msg);
        cloud_msg.header.stamp = this->scan_stamp;
        cloud_msg.header.frame_id = this->map_frame;

        this->submap_pub.publish(cloud_msg);
    }
}

void OdomNode::setPoseCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
    std::cout << "[OdomNode::setPoseCB]: Received request to set robot pose\n" << msg->pose.pose << "\n";

    // Set initial pose to requested pose
    this->initial_position_ = Eigen::Vector3f(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    this->initial_orientation_ =
        Eigen::Quaternionf(msg->pose.pose.orientation.w, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);

    std::cout << "[OdomNode::setPoseCB]: Initial pose set. Resetting DLO...\n";

    this->dlo_initialized = false;
}

/**
 * Publish Pose
 **/

void OdomNode::publishPose()
{
    // Sign flip check
    static Eigen::Quaternionf q_diff{ 1., 0., 0., 0. };
    static Eigen::Quaternionf q_last{ 1., 0., 0., 0. };

    q_diff = q_last.conjugate() * this->rotq;

    // If q_diff has negative real part then there was a sign flip
    if (q_diff.w() < 0)
    {
        this->rotq.w() = -this->rotq.w();
        this->rotq.vec() = -this->rotq.vec();
    }

    q_last = this->rotq;

    nav_msgs::Odometry odom_msg;

    odom_msg.pose.pose.position.x = this->pose[0];
    odom_msg.pose.pose.position.y = this->pose[1];
    odom_msg.pose.pose.position.z = this->pose[2];

    odom_msg.pose.pose.orientation.w = this->rotq.w();
    odom_msg.pose.pose.orientation.x = this->rotq.x();
    odom_msg.pose.pose.orientation.y = this->rotq.y();
    odom_msg.pose.pose.orientation.z = this->rotq.z();

    odom_msg.header.stamp = this->scan_stamp;
    odom_msg.header.frame_id = this->map_frame;
    odom_msg.child_frame_id = this->baselink_frame;
    this->lidar_odom_pub.publish(odom_msg);

    geometry_msgs::PoseStamped pose_msg;

    pose_msg.header.stamp = this->scan_stamp;
    pose_msg.header.frame_id = this->map_frame;

    pose_msg.pose.position.x = this->pose[0];
    pose_msg.pose.position.y = this->pose[1];
    pose_msg.pose.position.z = this->pose[2];

    pose_msg.pose.orientation.w = this->rotq.w();
    pose_msg.pose.orientation.x = this->rotq.x();
    pose_msg.pose.orientation.y = this->rotq.y();
    pose_msg.pose.orientation.z = this->rotq.z();

    this->pose_pub.publish(pose_msg);
}

/**
 * Publish Transform
 **/

void OdomNode::publishTransform()
{
    static tf::TransformListener tf_listener;
    tf::StampedTransform transform;
    try
    {
        tf_listener.waitForTransform(this->odom_frame, this->baselink_frame, this->scan_stamp, ros::Duration(0.5));
        tf_listener.lookupTransform(this->odom_frame, this->baselink_frame, this->scan_stamp, transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_DEBUG_STREAM("Cannot get transform [" << this->odom_frame << "] -> [" << this->baselink_frame << "] Exception: " << ex.what());
        return;
    }

    Eigen::Isometry3d odom_2_bl;
    tf::transformTFToEigen(transform, odom_2_bl);

    const Eigen::Isometry3d map_to_baselink(this->T.cast<double>());
    Eigen::Isometry3d map_2_odom = map_to_baselink * odom_2_bl.inverse();
    Eigen::Quaterniond normalized_quat(map_2_odom.rotation());
    normalized_quat.normalize();
    map_2_odom.matrix().topLeftCorner(3, 3) = normalized_quat.toRotationMatrix();

    const geometry_msgs::Pose map_2_odom_tf = tf2::toMsg(map_2_odom);

    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transform_stamped;

    transform_stamped.header.stamp = this->scan_stamp;
    transform_stamped.header.frame_id = this->map_frame;
    transform_stamped.child_frame_id = this->odom_frame;

    transform_stamped.transform.translation.x = map_2_odom_tf.position.x;
    transform_stamped.transform.translation.y = map_2_odom_tf.position.y;
    transform_stamped.transform.translation.z = map_2_odom_tf.position.z;
    transform_stamped.transform.rotation = map_2_odom_tf.orientation;

    br.sendTransform(transform_stamped);
}

/**
 * Publish Keyframe Pose and Scan
 **/

void OdomNode::publishKeyframe()
{
    // Publish keyframe pose
    nav_msgs::Odometry kf_odom;

    kf_odom.header.stamp = this->scan_stamp;
    kf_odom.header.frame_id = this->map_frame;
    kf_odom.child_frame_id = this->baselink_frame;

    kf_odom.pose.pose.position.x = this->pose.x();
    kf_odom.pose.pose.position.y = this->pose.y();
    kf_odom.pose.pose.position.z = this->pose.z();

    kf_odom.pose.pose.orientation.w = this->rotq.w();
    kf_odom.pose.pose.orientation.x = this->rotq.x();
    kf_odom.pose.pose.orientation.y = this->rotq.y();
    kf_odom.pose.pose.orientation.z = this->rotq.z();

    this->kf_pub.publish(kf_odom);

    // Publish keyframe scan
    if (!this->keyframe_cloud->points.empty() && this->keyframe_pub.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud2 keyframe_cloud_ros;
        pcl::toROSMsg(*this->keyframe_cloud, keyframe_cloud_ros);
        keyframe_cloud_ros.header.stamp = this->scan_stamp;
        keyframe_cloud_ros.header.frame_id = this->map_frame;
        this->keyframe_pub.publish(keyframe_cloud_ros);
    }
}

void OdomNode::publishAllKeyframes()
{
    geometry_msgs::PoseArray keyframe_pose_array;
    keyframe_pose_array.header.stamp = ros::Time::now();
    keyframe_pose_array.header.frame_id = this->map_frame;

    for (auto kf_pair : this->keyframes->frames())
    {
        geometry_msgs::Pose pose;

        pose.position.x = kf_pair.second.pose.translation().x();
        pose.position.y = kf_pair.second.pose.translation().y();
        pose.position.z = kf_pair.second.pose.translation().z();

        Eigen::Quaternionf quat(kf_pair.second.pose.rotation());

        pose.orientation.w = quat.w();
        pose.orientation.x = quat.x();
        pose.orientation.y = quat.y();
        pose.orientation.z = quat.z();

        keyframe_pose_array.poses.push_back(pose);
    }

    this->kf_array_pub.publish(keyframe_pose_array);
}

/**
 * Preprocessing
 **/

void OdomNode::preprocessPoints()
{
    // Original Scan
    *this->original_scan = *this->current_scan;

    // Remove NaNs
    std::vector<int> idx;
    this->current_scan->is_dense = false;
    pcl::removeNaNFromPointCloud(*this->current_scan, *this->current_scan, idx);

    // Crop Box Filter
    if (this->crop_use_)
    {
        this->crop.setInputCloud(this->current_scan);
        this->crop.filter(*this->current_scan);
    }

    // Voxel Grid Filter
    if (this->vf_scan_use_)
    {
        this->vf_scan.setInputCloud(this->current_scan);
        this->vf_scan.filter(*this->current_scan);
    }
}

/**
 * Initialize Input Target
 **/

void OdomNode::initializeInputTarget()
{
    this->prev_frame_stamp = this->curr_frame_stamp;

    // Convert ros message
    this->target_cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    this->target_cloud = this->current_scan;
    this->gicp_s2s.setInputTarget(this->target_cloud);
    this->gicp_s2s.calculateTargetCovariances();

    // initialize keyframes
    pcl::PointCloud<PointType>::Ptr first_keyframe(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*this->target_cloud, *first_keyframe, this->T);

    // voxelization for submap
    if (this->vf_submap_use_)
    {
        this->vf_submap.setInputCloud(first_keyframe);
        this->vf_submap.filter(*first_keyframe);
    }

    // keep history of keyframes
    *this->keyframe_cloud = *first_keyframe;

    // compute kdtree and keyframe normals (use gicp_s2s input source as temporary storage because it will be overwritten by setInputSources())
    this->gicp_s2s.setInputSource(this->keyframe_cloud);
    this->gicp_s2s.calculateSourceCovariances();
}

/**
 * Set Input Sources
 **/

void OdomNode::setInputSources()
{
    // set the input source for the S2S gicp
    // this builds the KdTree of the source cloud
    // this does not build the KdTree for s2m because force_no_update is true
    this->gicp_s2s.setInputSource(this->current_scan);

    // set pcl::Registration input source for S2M gicp using custom NanoGICP function
    this->gicp.registerInputSource(this->current_scan);

    // now set the KdTree of S2M gicp using previously built KdTree
    this->gicp.source_kdtree_ = this->gicp_s2s.source_kdtree_;
    this->gicp.source_covs_.clear();
}

/**
 * Gravity Alignment
 **/

void OdomNode::gravityAlign()
{
    // get average acceleration vector for 1 second and normalize
    Eigen::Vector3f lin_accel = Eigen::Vector3f::Zero();
    const double then = ros::Time::now().toSec();
    int n = 0;
    while ((ros::Time::now().toSec() - then) < 1.)
    {
        lin_accel[0] += this->imu_meas.lin_accel.x;
        lin_accel[1] += this->imu_meas.lin_accel.y;
        lin_accel[2] += this->imu_meas.lin_accel.z;
        ++n;
    }
    lin_accel[0] /= n;
    lin_accel[1] /= n;
    lin_accel[2] /= n;

    // normalize
    double lin_norm = sqrt(pow(lin_accel[0], 2) + pow(lin_accel[1], 2) + pow(lin_accel[2], 2));
    lin_accel[0] /= lin_norm;
    lin_accel[1] /= lin_norm;
    lin_accel[2] /= lin_norm;

    // define gravity vector (assume point downwards)
    Eigen::Vector3f grav;
    grav << 0, 0, 1;

    // calculate angle between the two vectors
    Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(lin_accel, grav);

    // normalize
    grav_q.normalize();

    // rpy
    Eigen::Vector3f euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
    float pitch = euler[1] * (180.0 / M_PI);
    float roll = euler[2] * (180.0 / M_PI);
    float yaw = this->rotq.toRotationMatrix().eulerAngles(2, 1, 0)[0];

    std::cout << "done" << std::endl;
    std::cout << "  Roll [deg]: " << roll << std::endl;
    std::cout << "  Pitch [deg]: " << pitch << std::endl << std::endl;

    // set gravity aligned roll/pitch
    this->rotq = grav_q;
    this->rotSO3 = this->rotq.toRotationMatrix();
    this->T.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
    this->T_s2s.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
    this->T_s2s_prev.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
}

/**
 * Initialize 6DOF
 **/

void OdomNode::initializeDLO()
{
    // Calibrate IMU
    if (!this->imu_calibrated && this->imu_use_)
    {
        return;
    }

    // Gravity Align
    if (this->gravity_align_ && this->imu_use_ && this->imu_calibrated)
    {
        std::cout << "Aligning to gravity... ";
        std::cout.flush();
        this->gravityAlign();
    }

    std::cout.flush();

    // set known position
    this->pose = this->initial_position_;
    this->T.block(0, 3, 3, 1) = this->pose;
    this->T_s2s.block(0, 3, 3, 1) = this->pose;
    this->T_s2s_prev.block(0, 3, 3, 1) = this->pose;

    // set known orientation
    this->rotq = this->initial_orientation_;
    this->rotSO3 = this->rotq.toRotationMatrix();
    this->T.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
    this->T_s2s.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
    this->T_s2s_prev.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();

    if (this->keyframes->size() > 0)
    {
        std::cout << "[OdomNode::initializeDLO]: Aligning scan to map and initializing robot pose...\n";

        // Validate initial pose against map data
        OptionalValue<Eigen::Isometry3f> corrected_initial_pose = this->validatePose(Eigen::Isometry3f(this->T), *this->current_scan);

        if (corrected_initial_pose)
        {
            this->pose = corrected_initial_pose().translation();
            this->rotq = Eigen::Quaternionf(corrected_initial_pose().rotation());

            // Set position
            this->initial_position_ = this->pose;
            this->T.block(0, 3, 3, 1) = this->pose;
            this->T_s2s.block(0, 3, 3, 1) = this->pose;
            this->T_s2s_prev.block(0, 3, 3, 1) = this->pose;

            // Set orientation
            this->initial_orientation_ = this->rotq;
            this->T.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
            this->T_s2s.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();
            this->T_s2s_prev.block(0, 0, 3, 3) = this->rotq.toRotationMatrix();

            std::cout << "[OdomNode::initializeDLO]: Set initial pose in current map [" << this->map_name << "] with position [" << this->pose.transpose()
                      << "] and orientation [" << this->rotq.coeffs().transpose() << "]\n";
        }
        else
        {
            std::cout << "[OdomNode::initializeDLO]: Could not validate initial pose in current map [" << this->map_name
                      << "]. Initialization failed. Will attempt again...\n";
            return;
        }
    }

    if (this->trajectory_optimization_use_)
    {
        // Add initial pose as prior to gtsam graph
        gtsam::Pose3 prior_pose(this->T.cast<double>());
        gtsam::noiseModel::Diagonal::shared_ptr prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        this->graph.add(gtsam::PriorFactor<gtsam::Pose3>(this->gtsam_keyframe_id, prior_pose, prior_noise));

        this->initial_guess.insert(0, prior_pose);

        this->gtsam_kf_id_map.insert({ 0, 0 });
        ++this->gtsam_keyframe_id;
    }

    this->trajectory.push_back(Eigen::Isometry3f(this->T));

    this->dlo_initialized = true;
    std::cout << "DLO initialized! Starting localization..." << std::endl;
}

/**
 * ICP Point Cloud Callback
 **/

void OdomNode::icpCB(const sensor_msgs::PointCloud2ConstPtr& pc)
{
    double then = ros::Time::now().toSec();
    this->scan_stamp = pc->header.stamp;
    this->curr_frame_stamp = pc->header.stamp.toSec();

    // If there are too few points in the pointcloud, try again
    this->current_scan = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*pc, *(this->current_scan));
    if (this->current_scan->points.size() < this->gicp_min_num_points_)
    {
        ROS_WARN("Low number of points!");
        return;
    }

    // Unrotate point cloud to align with base-link
    pcl::transformPointCloud(*(this->current_scan), *(this->current_scan), baselink_tf_lidar_.matrix());

    // Preprocess points
    this->preprocessPoints();

    if (!this->dlo_initialized)
    {
        // DLO Initialization procedures (IMU calib, gravity align)
        this->initializeDLO();
    }
    else
    {
        // Compute Metrics
        this->computeMetrics();

        // Set Adaptive Parameters
        if (this->adaptive_params_use_)
        {
            this->setAdaptiveParams();
        }

        // Set initial frame as target
        if (this->target_cloud == nullptr)
        {
            this->initializeInputTarget();
            this->updateKeyframes();
            return;
        }

        // Set source frame
        this->source_cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        this->source_cloud = this->current_scan;

        // Set new frame as input source for both gicp objects
        this->setInputSources();

        // Get the next pose via IMU + S2S + S2M
        this->getNextPose();

        // Update current keyframe poses and map
        this->updateKeyframes();

        if (this->trajectory_optimization_use_)
        {
            // Optimize trajectory
            this->optimizeTrajectory();
        }
    }

    // Update next time stamp
    this->prev_frame_stamp = this->curr_frame_stamp;

    // Update some statistics
    this->comp_times.push_back(ros::Time::now().toSec() - then);

    // Publish stuff to ROS
    this->publishToROS();
}

void OdomNode::optimizeTrajectory()
{
    // Add latest pose to trajectory
    this->trajectory.push_back(Eigen::Isometry3f(this->T));

    this->initial_guess.insert(this->gtsam_keyframe_id, gtsam::Pose3(this->T.cast<double>()));

    const int curr_pose_idx = this->trajectory.size() - 1;
    const int prev_pose_idx = this->trajectory.size() - 2;

    Eigen::Isometry3f factor_pose = this->trajectory[prev_pose_idx].inverse() * this->trajectory[curr_pose_idx];
    gtsam::noiseModel::Diagonal::shared_ptr noiseModel = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);
    this->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(this->gtsam_keyframe_id - 1, this->gtsam_keyframe_id, gtsam::Pose3(factor_pose.matrix().cast<double>()),
                                                       noiseModel));

    this->gtsam_keyframe_id++;

    // Update solver
    this->isam_solver.update(this->graph, this->initial_guess);
    this->isam_solver.update();

    this->initial_guess.clear();
    this->graph.resize(0);

    // Optimize and retrieve trajectory from solver
    this->optimized_estimate = this->isam_solver.calculateEstimate();

    // Update trajectory
    for (unsigned int pose_id = 0; pose_id < this->optimized_estimate.size(); pose_id++)
    {
        // Correct trajectory poses
        gtsam::Pose3 optimized_pose = this->optimized_estimate.at<gtsam::Pose3>(pose_id);
        trajectory[pose_id] = Eigen::Isometry3f(optimized_pose.matrix().cast<float>());

        if (this->gtsam_kf_id_map.find(pose_id) != this->gtsam_kf_id_map.end())
        {
            // Correct keyframe poses
            const int kf_id = this->gtsam_kf_id_map.at(pose_id);
            this->keyframes->setFramePose(kf_id, trajectory[pose_id]);
        }
    }

    Eigen::Isometry3f current_pose(this->optimized_estimate.at<gtsam::Pose3>(this->optimized_estimate.size() - 1).matrix().cast<float>());
    this->T = current_pose.matrix();
    this->pose = current_pose.translation();
    this->rotq = Eigen::Quaternionf(current_pose.rotation());
    this->rotSO3 = current_pose.rotation();
}

/**
 * Odom Callback
 **/

void OdomNode::odomCB(const nav_msgs::Odometry::ConstPtr& odom)
{
    if (!this->odom_use_)
    {
        return;
    }

    this->odom_meas.stamp = odom->header.stamp.toSec();

    this->odom_meas.lin_vel.x = odom->twist.twist.linear.x;
    this->odom_meas.lin_vel.y = odom->twist.twist.linear.y;
    this->odom_meas.lin_vel.z = odom->twist.twist.linear.z;

    this->odom_meas.ang_vel.x = odom->twist.twist.angular.x;
    this->odom_meas.ang_vel.y = odom->twist.twist.angular.y;
    this->odom_meas.ang_vel.z = odom->twist.twist.angular.z;

    this->mtx_odom.lock();
    this->odom_buffer.push_front(this->odom_meas);
    this->mtx_odom.unlock();
}

/**
 * IMU Callback
 **/

void OdomNode::imuCB(const sensor_msgs::Imu::ConstPtr& imu)
{
    if (!this->imu_use_)
    {
        return;
    }

    double ang_vel[3], lin_accel[3];

    // Get IMU samples
    ang_vel[0] = imu->angular_velocity.x;
    ang_vel[1] = imu->angular_velocity.y;
    ang_vel[2] = imu->angular_velocity.z;

    lin_accel[0] = imu->linear_acceleration.x;
    lin_accel[1] = imu->linear_acceleration.y;
    lin_accel[2] = imu->linear_acceleration.z;

    if (this->first_imu_time == 0.)
    {
        this->first_imu_time = imu->header.stamp.toSec();
    }

    // IMU calibration procedure - do for three seconds
    if (!this->imu_calibrated)
    {
        static int num_samples = 0;
        static bool print = true;

        if ((imu->header.stamp.toSec() - this->first_imu_time) < this->imu_calib_time_)
        {
            num_samples++;

            this->imu_bias.gyro.x += ang_vel[0];
            this->imu_bias.gyro.y += ang_vel[1];
            this->imu_bias.gyro.z += ang_vel[2];

            this->imu_bias.accel.x += lin_accel[0];
            this->imu_bias.accel.y += lin_accel[1];
            this->imu_bias.accel.z += lin_accel[2];

            if (print)
            {
                std::cout << "Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
                std::cout.flush();
                print = false;
            }
        }
        else
        {
            this->imu_bias.gyro.x /= num_samples;
            this->imu_bias.gyro.y /= num_samples;
            this->imu_bias.gyro.z /= num_samples;

            this->imu_bias.accel.x /= num_samples;
            this->imu_bias.accel.y /= num_samples;
            this->imu_bias.accel.z /= num_samples;

            this->imu_calibrated = true;

            std::cout << "done" << std::endl;
            std::cout << "  Gyro biases [xyz]: " << this->imu_bias.gyro.x << ", " << this->imu_bias.gyro.y << ", " << this->imu_bias.gyro.z << std::endl
                      << std::endl;
        }
    }
    else
    {
        // Apply the calibrated bias to the new IMU measurements
        this->imu_meas.stamp = imu->header.stamp.toSec();

        this->imu_meas.ang_vel.x = ang_vel[0] - this->imu_bias.gyro.x;
        this->imu_meas.ang_vel.y = ang_vel[1] - this->imu_bias.gyro.y;
        this->imu_meas.ang_vel.z = ang_vel[2] - this->imu_bias.gyro.z;

        this->imu_meas.lin_accel.x = lin_accel[0];
        this->imu_meas.lin_accel.y = lin_accel[1];
        this->imu_meas.lin_accel.z = lin_accel[2];

        // Store into circular buffer
        this->mtx_imu.lock();
        this->imu_buffer.push_front(this->imu_meas);
        this->mtx_imu.unlock();
    }
}

/**
 * Get Next Pose
 **/

void OdomNode::getNextPose()
{
    //
    // FRAME-TO-FRAME PROCEDURE
    //

    // Align using IMU prior if available
    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>);

    if (this->imu_use_)
    {
        this->integrateIMU();
        this->gicp_s2s.align(*aligned, this->imu_SE3);
    }
    else if (this->odom_use_)
    {
        this->integrateOdom();
        this->gicp_s2s.align(*aligned, this->odom_SE3);
    }
    else
    {
        this->gicp_s2s.align(*aligned);
    }

    // Get the local S2S transform
    Eigen::Matrix4f T_S2S = this->gicp_s2s.getFinalTransformation();

    // Get the global S2S transform
    this->propagateS2S(T_S2S);

    // reuse covariances from s2s for s2m
    this->gicp.source_covs_ = this->gicp_s2s.source_covs_;

    // Swap source and target (which also swaps KdTrees internally) for next S2S
    this->gicp_s2s.swapSourceAndTarget();

    //
    // FRAME-TO-SUBMAP
    //

    // Get current global submap
    this->keyframes->buildSubmap(this->T_s2s.block(0, 3, 3, 1));
    if (this->keyframes->submapHasChanged())
    {
        // Set the current global submap as the target cloud
        this->gicp.setInputTarget(this->keyframes->submapCloud());

        // Set target cloud's normals as submap normals
        this->gicp.setTargetCovariances(this->keyframes->submapNormals());
    }

    // Align with current submap with global S2S transformation as initial guess
    this->gicp.align(*aligned, this->T_s2s);

    // Get final transformation in global frame
    this->T = this->gicp.getFinalTransformation();

    // Update the S2S transform for next propagation
    this->T_s2s_prev = this->T;

    // Update next global pose
    // Both source and target clouds are in the global frame now, so tranformation is global
    this->propagateS2M();

    // Set next target cloud as current source cloud
    *this->target_cloud = *this->source_cloud;
}

OptionalValue<Eigen::Isometry3f> OdomNode::validatePose(const Eigen::Isometry3f& pose_guess, const pcl::PointCloud<PointType>& scan)
{
    std::cout << "[OdomNode::validatePose]: Seed position [" << pose_guess.translation().transpose() << "]\n";

    // Initialize ICP object
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(40.0);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Set current scan as source cloud
    pcl::PointCloud<PointType>::Ptr current_scan(new pcl::PointCloud<PointType>(scan));
    pcl::PointCloud<PointType>::Ptr transformed_current_scan(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*current_scan, *transformed_current_scan, pose_guess.matrix());
    icp.setInputSource(transformed_current_scan);

    // Extract submap near initial guess
    this->keyframes->buildSubmap(pose_guess.translation());
    pcl::PointCloud<PointType>::Ptr submap_cloud(new pcl::PointCloud<PointType>);
    *submap_cloud = *this->keyframes->submapCloud();

    // Set submap cloud as target cloud
    icp.setInputTarget(submap_cloud);

    // Align scan to map
    pcl::PointCloud<PointType>::Ptr aligned_scan(new pcl::PointCloud<PointType>);
    icp.align(*aligned_scan);

    std::cout << "[OdomNode::validatePose]: ICP Fitness Score: [" << icp.getFitnessScore() << "]\n";

    // Check convergence
    if (!icp.hasConverged())
    {
        std::cout << "[OdomNode::validatePose]: ICP has not converged. Cannot validate initial pose\n";
        return {};
    }

    // Extract pose correction transform
    Eigen::Isometry3f pose_correction = Eigen::Isometry3f(icp.getFinalTransformation());

    std::cout << "[OdomNode::validatePose]: Pose correction [" << pose_correction.translation().transpose() << "]\n";

    // Compute pose in map frame
    Eigen::Isometry3f corrected_pose = pose_guess * pose_correction;

    std::cout << "[OdomNode::validatePose]: Corrected position [" << corrected_pose.translation().transpose() << "]\n";

    // Check if correction is within allowed distance bounds
    float distance = pose_correction.translation().norm();
    if (distance > this->max_initialization_distance_threshold_m_)
    {
        std::cout << "[OdomNode::validatePose]: Aligned initial point is more than " << this->max_initialization_distance_threshold_m_
                  << " m threshold from initial guess. Distance: [" << distance << "]\n";
        return {};
    }

    // Check if correction is within allowed angle bounds
    float angle = 2.0 * std::abs(std::acos(Eigen::Quaternionf(pose_correction.rotation()).w()));
    if (angle > this->max_initialization_angle_threshold_rad_)
    {
        std::cout << "[OdomNode::validatePose]: Aligned initial point is more than " << this->max_initialization_angle_threshold_rad_
                  << " rad threshold from initial guess. Angle: [" << angle << "]\n";
        return {};
    }

    std::cout << "[OdomNode::validatePose]: Distance correction [" << distance << "], angle correction [" << angle << "]\n";

    return { corrected_pose };
}

/**
 * Integrate Odom
 **/

void OdomNode::integrateOdom()
{
    // Extract odom data between the two frames
    std::vector<OdomMeas> odom_frames;

    for (const auto& o : this->odom_buffer)
    {
        // Odom data between two frames is when:
        //   current frame's timestamp minus odom timestamp is positive
        //   previous frame's timestamp minus odom timestamp is negative
        double curr_frame_odom_dt = this->curr_frame_stamp - o.stamp;
        double prev_frame_odom_dt = this->prev_frame_stamp - o.stamp;

        if (curr_frame_odom_dt >= 0. && prev_frame_odom_dt <= 0.)
        {
            odom_frames.push_back(o);
        }
    }

    // Sort measurements by time
    std::sort(odom_frames.begin(), odom_frames.end(), this->comparatorMeas);

    // Relative Odom integration of gyro and accelerometer
    double curr_odom_stamp = 0.;
    double prev_odom_stamp = 0.;
    double dt;

    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();

    for (uint32_t i = 0; i < odom_frames.size(); ++i)
    {
        if (prev_odom_stamp == 0.)
        {
            prev_odom_stamp = odom_frames[i].stamp;
            continue;
        }

        // Calculate difference in odom measurement times IN SECONDS
        curr_odom_stamp = odom_frames[i].stamp;
        dt = curr_odom_stamp - prev_odom_stamp;
        prev_odom_stamp = curr_odom_stamp;

        // Relative gyro propagation quaternion dynamics
        Eigen::Quaternionf qq = q;
        q.w() -= 0.5 * (qq.x() * odom_frames[i].ang_vel.x + qq.y() * odom_frames[i].ang_vel.y + qq.z() * odom_frames[i].ang_vel.z) * dt;
        q.x() += 0.5 * (qq.w() * odom_frames[i].ang_vel.x - qq.z() * odom_frames[i].ang_vel.y + qq.y() * odom_frames[i].ang_vel.z) * dt;
        q.y() += 0.5 * (qq.z() * odom_frames[i].ang_vel.x + qq.w() * odom_frames[i].ang_vel.y - qq.x() * odom_frames[i].ang_vel.z) * dt;
        q.z() += 0.5 * (qq.x() * odom_frames[i].ang_vel.y - qq.y() * odom_frames[i].ang_vel.x + qq.w() * odom_frames[i].ang_vel.z) * dt;

        Eigen::Vector3f tt = t;
        t.x() += tt.x() + odom_frames[i].lin_vel.x * dt;
        t.y() += tt.y() + odom_frames[i].lin_vel.y * dt;
        t.z() += tt.z() + odom_frames[i].lin_vel.z * dt;
    }

    // Normalize quaternion
    double norm = sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    q.w() /= norm;
    q.x() /= norm;
    q.y() /= norm;
    q.z() /= norm;

    // Store Odom guess
    this->odom_SE3 = Eigen::Matrix4f::Identity();
    this->odom_SE3.block<3, 3>(0, 0) = q.toRotationMatrix();
    this->odom_SE3.block<3, 1>(0, 3) = t;
}

/**
 * Integrate IMU
 **/

void OdomNode::integrateIMU()
{
    // Extract IMU data between the two frames
    std::vector<ImuMeas> imu_frame;

    for (const auto& i : this->imu_buffer)
    {
        // IMU data between two frames is when:
        //   current frame's timestamp minus imu timestamp is positive
        //   previous frame's timestamp minus imu timestamp is negative
        double curr_frame_imu_dt = this->curr_frame_stamp - i.stamp;
        double prev_frame_imu_dt = this->prev_frame_stamp - i.stamp;

        if (curr_frame_imu_dt >= 0. && prev_frame_imu_dt <= 0.)
        {
            imu_frame.push_back(i);
        }
    }

    // Sort measurements by time
    std::sort(imu_frame.begin(), imu_frame.end(), this->comparatorMeas);

    // Relative IMU integration of gyro and accelerometer
    double curr_imu_stamp = 0.;
    double prev_imu_stamp = 0.;
    double dt;

    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();

    for (uint32_t i = 0; i < imu_frame.size(); ++i)
    {
        if (prev_imu_stamp == 0.)
        {
            prev_imu_stamp = imu_frame[i].stamp;
            continue;
        }

        // Calculate difference in imu measurement times IN SECONDS
        curr_imu_stamp = imu_frame[i].stamp;
        dt = curr_imu_stamp - prev_imu_stamp;
        prev_imu_stamp = curr_imu_stamp;

        // Relative gyro propagation quaternion dynamics
        Eigen::Quaternionf qq = q;
        q.w() -= 0.5 * (qq.x() * imu_frame[i].ang_vel.x + qq.y() * imu_frame[i].ang_vel.y + qq.z() * imu_frame[i].ang_vel.z) * dt;
        q.x() += 0.5 * (qq.w() * imu_frame[i].ang_vel.x - qq.z() * imu_frame[i].ang_vel.y + qq.y() * imu_frame[i].ang_vel.z) * dt;
        q.y() += 0.5 * (qq.z() * imu_frame[i].ang_vel.x + qq.w() * imu_frame[i].ang_vel.y - qq.x() * imu_frame[i].ang_vel.z) * dt;
        q.z() += 0.5 * (qq.x() * imu_frame[i].ang_vel.y - qq.y() * imu_frame[i].ang_vel.x + qq.w() * imu_frame[i].ang_vel.z) * dt;
    }

    // Normalize quaternion
    double norm = sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    q.w() /= norm;
    q.x() /= norm;
    q.y() /= norm;
    q.z() /= norm;

    // Store IMU guess
    this->imu_SE3 = Eigen::Matrix4f::Identity();
    this->imu_SE3.block(0, 0, 3, 3) = q.toRotationMatrix();
}

/**
 * Propagate S2S Alignment
 **/

void OdomNode::propagateS2S(Eigen::Matrix4f T)
{
    this->T_s2s = this->T_s2s_prev * T;
    this->T_s2s_prev = this->T_s2s;

    this->pose_s2s << this->T_s2s(0, 3), this->T_s2s(1, 3), this->T_s2s(2, 3);
    this->rotSO3_s2s << this->T_s2s(0, 0), this->T_s2s(0, 1), this->T_s2s(0, 2), this->T_s2s(1, 0), this->T_s2s(1, 1), this->T_s2s(1, 2), this->T_s2s(2, 0),
        this->T_s2s(2, 1), this->T_s2s(2, 2);

    this->rotq_s2s = Eigen::Quaternionf(this->rotSO3_s2s).normalized();
}

/**
 * Propagate S2M Alignment
 **/

void OdomNode::propagateS2M()
{
    this->pose << this->T(0, 3), this->T(1, 3), this->T(2, 3);
    this->rotSO3 << this->T(0, 0), this->T(0, 1), this->T(0, 2), this->T(1, 0), this->T(1, 1), this->T(1, 2), this->T(2, 0), this->T(2, 1), this->T(2, 2);
    this->rotq = Eigen::Quaternionf(this->rotSO3).normalized();
}

/**
 * Compute Metrics
 **/

void OdomNode::computeMetrics()
{
    this->computeSpaciousness();
}

/**
 * Compute Spaciousness of Current Scan
 **/

void OdomNode::computeSpaciousness()
{
    // compute range of points
    std::vector<float> ds;

    for (int i = 0; i <= this->current_scan->points.size(); i++)
    {
        float d = std::sqrt(pow(this->current_scan->points[i].x, 2) + pow(this->current_scan->points[i].y, 2) + pow(this->current_scan->points[i].z, 2));
        ds.push_back(d);
    }

    // median
    std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
    float median_curr = ds[ds.size() / 2];
    static float median_prev = median_curr;
    float median_lpf = 0.95 * median_prev + 0.05 * median_curr;
    median_prev = median_lpf;

    // push
    this->metrics.spaciousness.push_back(median_lpf);
}

/**
 * Update keyframes
 **/

void OdomNode::updateKeyframes()
{
    // calculate difference in pose and rotation to all poses in trajectory
    float closest_d = std::numeric_limits<float>::infinity();
    int closest_idx = 0;
    int keyframes_idx = 0;

    int num_nearby = 0;

    for (const auto kf_pair : this->keyframes->frames())
    {
        float delta_d = (kf_pair.second.pose.translation() - this->pose).norm();

        // count the number nearby current pose
        if (delta_d <= this->keyframe_thresh_dist_ * 1.5)
        {
            ++num_nearby;
        }

        // store into variable
        if (delta_d < closest_d)
        {
            closest_d = delta_d;
            closest_idx = kf_pair.first;
        }
    }

    // update keyframe
    bool newKeyframe = false;

    if (num_nearby == 0)
    {
        newKeyframe = true;
    }
    else
    {
        // get closest pose and corresponding rotation
        Eigen::Quaternionf closest_pose_r(this->keyframes->getFramePose(closest_idx).rotation());

        // calculate difference in orientation
        Eigen::Quaternionf dq = this->rotq * (closest_pose_r.inverse());

        float theta_rad = 2. * atan2(sqrt(pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2)), dq.w());
        float theta_deg = theta_rad * (180.0 / M_PI);

        if (abs(closest_d) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_)
        {
            newKeyframe = true;
        }
        if (abs(closest_d) <= this->keyframe_thresh_dist_)
        {
            newKeyframe = false;
        }
        if (abs(closest_d) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1)
        {
            newKeyframe = true;
        }
    }

    if (newKeyframe)
    {
        // voxelization for submap
        if (this->vf_submap_use_)
        {
            this->vf_submap.setInputCloud(this->current_scan);
            this->vf_submap.filter(*this->current_scan);
        }

        this->gicp_s2s.setInputSource(this->current_scan);
        this->gicp_s2s.calculateSourceCovariances();

        Keyframe<int> kf(this->keyframes->size(), this->scan_stamp.toSec());
        kf.pose = Eigen::Isometry3f(this->T);
        kf.cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>(*this->current_scan));
        kf.normals = this->gicp_s2s.getSourceCovariances();

        // update keyframe map
        if (!this->keyframes->addFrame(kf))
        {
            std::cout << "[OdomNode::updateKeyframes]: Could not add keyframe with id [" << kf.id << "]\n";
        }

        *this->keyframe_cloud = *kf.getTransformedCloud();

        if (this->trajectory_optimization_use_)
        {
            // update keyframe id gtsam frame id association map
            this->gtsam_kf_id_map.insert({ this->gtsam_keyframe_id, kf.id });
        }

        this->publishKeyframe();
    }
}

/**
 * Set Adaptive Parameters
 **/

void OdomNode::setAdaptiveParams()
{
    // Set Keyframe Thresh from Spaciousness Metric
    if (this->metrics.spaciousness.back() > 20.0)
    {
        this->keyframe_thresh_dist_ = 10.0;
    }
    else if (this->metrics.spaciousness.back() > 10.0 && this->metrics.spaciousness.back() <= 20.0)
    {
        this->keyframe_thresh_dist_ = 5.0;
    }
    else if (this->metrics.spaciousness.back() > 5.0 && this->metrics.spaciousness.back() <= 10.0)
    {
        this->keyframe_thresh_dist_ = 1.0;
    }
    else if (this->metrics.spaciousness.back() <= 5.0)
    {
        this->keyframe_thresh_dist_ = 0.5;
    }

    // set concave hull alpha
    this->keyframes->setCvHullAlpha(this->keyframe_thresh_dist_);
}

bool OdomNode::saveCallback(er_file_io_msgs::HandleFile::Request& request, er_file_io_msgs::HandleFile::Response& response)
{
    if (request.file_path_and_name.empty())
    {
        std::cout << "[OdomNode::saveCallback]: Cannot save map with empty name string\n";
        return true;
    }

    this->map_name = this->map_prefix + "-" + request.file_path_and_name;
    saveMapFramesToDisk(this->keyframes->frames(), this->map_directory + "/" + this->map_name);
    return true;
}

bool OdomNode::loadCallback(er_file_io_msgs::HandleFile::Request& request, er_file_io_msgs::HandleFile::Response& response)
{
    // KeyframeMap<int> frames = loadMapFramesFromDisk(request.file_path_and_name);
    // this->keyframes = std::make_shared<Frames>(this->submap_params, frames);
    std::cout << "[OdomNode::loadCallback]: Map load feature is currently disabled\n";
    return false;
}

bool OdomNode::resetCallback(er_nav_msgs::SetLocalizationState::Request& request, er_nav_msgs::SetLocalizationState::Response& response)
{
    std::cout << "[OdomNode::resetCallback]: Requested new map [" << request.file_tag << "], current map [" << this->map_name << "] with ["
              << this->keyframes->size() << "] keyframes will be cleared\n";

    if (!request.file_tag.empty() && request.file_tag != this->map_name)
    {
        // Stop sensor data subscribers
        std::cout << "[OdomNode::resetCallback]: Stopping subscribers to sensor data ...\n";
        this->stopSubscribers();

        // Reset and init class variables
        // NOTE: This resets the current pose to origin and clears the keyframes
        this->init();

        // Load new map
        if (this->loadMap(request.file_tag))
        {
            std::cout << "[OdomNode::resetCallback]: Loaded new map [" << request.file_tag << "]\n";
        }
        else
        {
            std::cout << "[OdomNode::resetCallback]: Reset map\n";
        }

        // Start sensor data subscribers
        std::cout << "[OdomNode::resetCallback]: Starting subscribers to sensor data ...\n";
        this->startSubscribers();

        response.success = true;
    }
    else if (request.file_tag.empty())
    {
        // Stop sensor data subscribers
        std::cout << "[OdomNode::resetCallback]: Stopping subscribers to sensor data ...\n";
        this->stopSubscribers();

        // Reset map name
        this->map_name.clear();

        // Reset and init class variables
        // NOTE: This resets the current pose to origin and clears the keyframes
        this->init();

        std::cout << "[OdomNode::resetCallback]: Cleared map and recording new map\n";

        // Start sensor data subscribers
        std::cout << "[OdomNode::resetCallback]: Starting subscribers to sensor data ...\n";
        this->startSubscribers();

        response.success = true;
    }
    else
    {
        std::cout << "Requested map [" << request.file_tag << "] is already the loaded map. Doing nothing\n";
        response.success = true;
    }

    return true;
}

bool OdomNode::getCurrentWaypointCallback(er_nav_msgs::GetCurrentWaypoint::Request& request, er_nav_msgs::GetCurrentWaypoint::Response& response)
{
    return true;
}

bool OdomNode::getWaypointByIdCallback(er_nav_msgs::GetWaypointById::Request& request, er_nav_msgs::GetWaypointById::Response& response)
{
    return true;
}
}  // namespace dlo
