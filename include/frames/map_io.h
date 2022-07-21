#pragma once

#include <frames/frames.h>
#include <frames/point_types.h>
#include <nano_gicp/nano_gicp.hpp>

#include <pcl/common/io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>
#include <fstream>
#include <chrono>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/make_shared.hpp>

#include <Eigen/Geometry>

// #include <frames/progbar.h>

namespace dlo
{
constexpr int k_file_index_chars = 6;

static Eigen::Isometry3f toEigenPose(const PoseType& pose)
{
    return Eigen::Isometry3f(pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw).matrix());
}

static PoseType toPclPose(const Eigen::Isometry3f& pose)
{
    PoseType pcl_pose;

    pcl_pose.x = pose.translation().x();
    pcl_pose.y = pose.translation().y();
    pcl_pose.z = pose.translation().z();

    Eigen::Vector3f rpy = pose.rotation().eulerAngles(0, 1, 2);

    pcl_pose.roll = rpy.x();
    pcl_pose.pitch = rpy.y();
    pcl_pose.yaw = rpy.z();

    return pcl_pose;
}

static PointType toPclPoint(const Eigen::Isometry3f& pose)
{
    PointType pcl_point;

    pcl_point.x = pose.translation().x();
    pcl_point.y = pose.translation().y();
    pcl_point.z = pose.translation().z();

    return pcl_point;
}

static void fillNormals(KeyframeMap<int>& keyframes)
{
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    for (auto& kf_pair : keyframes)
    {
        gicp.setInputSource(kf_pair.second.cloud);
        gicp.calculateSourceCovariances();
        kf_pair.second.normals = gicp.getSourceCovariances();
    }
}

static void saveMapFramesToDisk(const KeyframeMap<int>& frames, const std::string& dir_path)
{
    std::cout << "****************************************************[saveMapFramesToDisk]****************************************************\n";

    std::string suffix = "";
    if (!boost::filesystem::exists(dir_path))
    {
        std::cout << "Creating directory [" << dir_path << "]\n";
    }
    else
    {
        double now = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1e9;
        suffix += "." + std::to_string(now);
    }

    boost::filesystem::create_directory(dir_path + suffix);

    std::cout << "[saveMapFramesToDisk]: Saving map files to [" << dir_path + suffix << "] ...\n";

    PointCloud::Ptr frame_positions(new PointCloud);
    PoseCloud::Ptr frame_poses(new PoseCloud);
    PointCloud::Ptr global_map(new PointCloud);

    // utility::progbar bar(std::cout, 80, '=', ' ');
    for (auto kf_pair : frames)
    {
        std::ostringstream padded_index;
        padded_index << std::internal << std::setfill('0') << std::setw(k_file_index_chars) << std::to_string(kf_pair.first);

        pcl::io::savePCDFileBinary(dir_path + suffix + "/" + padded_index.str() + "-key_frame_points.pcd", *kf_pair.second.cloud);

        PoseType pcl_pose = toPclPose(kf_pair.second.pose);
        pcl_pose.time = kf_pair.second.timestamp;

        frame_positions->push_back(toPclPoint(kf_pair.second.pose));
        frame_poses->push_back(pcl_pose);

        *global_map += *kf_pair.second.getTransformedCloud();

        // bar.update(kf_pair.first + 1, frames.size());
    }

    pcl::io::savePCDFileASCII(dir_path + suffix + "/key_frames_positions.pcd", *frame_positions);
    pcl::io::savePCDFileASCII(dir_path + suffix + "/key_frames_poses.pcd", *frame_poses);
    pcl::io::savePCDFileASCII(dir_path + suffix + "/full_key_frames_map.pcd", *global_map);

    std::cout << "[saveMapFramesToDisk]: Saving map files to [" << dir_path + suffix << "] completed\n";
    std::cout << "****************************************************[saveMapFramesToDisk]****************************************************\n";
}

static KeyframeMap<int> loadMapFramesFromDisk(const std::string& dir_path)
{
    std::cout << "****************************************************[loadMapFramesFromDisk]****************************************************\n";
    std::cout << "[loadMapFramesFromDisk]: Loading map files from [" << dir_path << "] ...\n";

    PointCloud::Ptr frame_positions(new PointCloud);
    PoseCloud::Ptr frame_poses(new PoseCloud);
    pcl::io::loadPCDFile(dir_path + "/key_frames_positions.pcd", *frame_positions);
    pcl::io::loadPCDFile(dir_path + "/key_frames_poses.pcd", *frame_poses);

    KeyframeMap<int> frames;
    // utility::progbar bar(std::cout, 80, '=', ' ');
    for (size_t i = 0; i < frame_positions->size(); ++i)
    {
        Keyframe<int> frame(i, frame_poses->at(i).time);
        frame.cloud = PointCloud::Ptr(new PointCloud);
        frame.pose = toEigenPose(frame_poses->at(i));

        std::ostringstream padded_index;
        padded_index << std::internal << std::setfill('0') << std::setw(k_file_index_chars) << std::to_string(i);

        pcl::io::loadPCDFile(dir_path + "/" + padded_index.str() + "-key_frame_points.pcd", *(frame.cloud));

        frames.insert({ i, frame });
        // bar.update(i + 1, frame_positions->size());
    }

    fillNormals(frames);

    std::cout << "[loadMapFramesFromDisk]: Loaded [" << frame_positions->size() << "] feature clouds\n";
    std::cout << "[loadMapFramesFromDisk]: Loading map from [" << dir_path << "] completed\n";
    std::cout << "****************************************************[loadMapFramesFromDisk]****************************************************\n";

    return frames;
}
}  // namespace dlo
