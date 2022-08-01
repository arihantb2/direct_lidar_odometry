#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>

#include <frames/point_types.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bimap.hpp>

namespace dlo
{
using PointCloud = pcl::PointCloud<PointType>;
using FrameNormals = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;
using PoseCloud = pcl::PointCloud<PoseType>;

template <typename T>
struct Keyframe
{
    Keyframe() : id{}
    {
        this->init();
    }
    Keyframe(const T& _id) : id{ _id }
    {
        this->init();
    }
    Keyframe(const T& _id, const std::string& uuid = "") : id{ _id }
    {
        this->init();

        if (!uuid.empty())
        {
            uuid_ = uuid;
        }
    }
    Keyframe(const T& _id, const double _timestamp, const std::string& uuid = "") : id{ _id }
    {
        this->init();
        this->timestamp = _timestamp;

        if (!uuid.empty())
        {
            uuid_ = uuid;
        }
    }
    Keyframe(const Keyframe& other)
    {
        this->id = other.id;
        this->uuid_ = other.uuid();
        this->timestamp = other.timestamp;
        this->pose = other.pose;
        this->cloud = PointCloud::Ptr(new PointCloud);
        if (other.cloud)
        {
            *this->cloud = *other.cloud;
            this->normals = other.normals;
        }
    }

    std::string uuid() const
    {
        return this->uuid_;
    }

    void init()
    {
        this->timestamp = 0.0;
        this->pose.setIdentity();
        this->cloud.reset();
        this->normals.clear();

        static boost::uuids::random_generator uuid_generator;
        this->uuid_ = boost::lexical_cast<std::string>(uuid_generator());
    }

    PointCloud::Ptr getTransformedCloud() const
    {
        PointCloud::Ptr tf_cloud(new PointCloud);

        if (this->cloud != nullptr)
        {
            pcl::transformPointCloud(*this->cloud, *tf_cloud, this->pose.matrix());
        }

        return tf_cloud;
    }

    T id;
    double timestamp;
    Eigen::Isometry3f pose;
    PointCloud::Ptr cloud;
    FrameNormals normals;

private:
    std::string uuid_;
};

template <typename IdType = int>
using KeyframeList = std::vector<Keyframe<IdType>>;

template <typename IdType = int>
using KeyframeMap = std::map<IdType, Keyframe<IdType>>;

class Frames
{
public:
    struct SubmapParams
    {
        int submap_knn_;
        int submap_kcv_;
        int submap_kcc_;
    };

    Frames(const SubmapParams& params);

    Frames(const SubmapParams&, const KeyframeMap<int>& keyframes);

    ~Frames();

    void init();

    bool add(const Keyframe<int>& frame);

    Keyframe<int> frame(const unsigned int idx) const;

    Keyframe<int> frame(const std::string& uuid) const;

    KeyframeMap<int> frames() const;

    KeyframeMap<int> operator()() const;

    unsigned int size() const;

    bool empty() const;

    void setCvHullAlpha(double alpha);

    bool setFramePose(const unsigned int idx, const Eigen::Isometry3f& pose);

    std::pair<bool, int> getClosestFrameId(Eigen::Vector3f pose);

    std::pair<bool, Keyframe<int>> getClosestFrame(Eigen::Vector3f pose);

    void buildMap();

    void buildSubmap(Eigen::Vector3f pose);

    void resetSubmapStates();

    bool submapHasChanged() const;

    PointCloud::Ptr submapCloud() const;

    PointCloud::Ptr getSubmapCloudOneTime(Eigen::Vector3f pose);

    FrameNormals submapNormals() const;

    PointCloud::Ptr mapCloud();

    PointCloud::Ptr mapCloudDS(const double leaf_size);

private:
    void computeConvexHull();

    void computeConcaveHull();

    void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frame_indices);

    KeyframeMap<int> frames_;
    boost::bimap<int, std::string> id_uuid_map_;

    pcl::ConvexHull<PointType> convex_hull_;
    pcl::ConcaveHull<PointType> concave_hull_;
    std::vector<int> keyframe_convex_;
    std::vector<int> keyframe_concave_;

    std::vector<int> submap_kf_idx_curr_;
    std::vector<int> submap_kf_idx_prev_;
    std::atomic<bool> submap_has_changed_;

    PointCloud::Ptr submap_cloud_;
    FrameNormals submap_normals_;

    bool map_changed_;
    PointCloud::Ptr map_cloud_;

    pcl::VoxelGrid<PointType> voxelgrid;

    SubmapParams params_;
};
}  // namespace dlo
