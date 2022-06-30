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

    Keyframe(T _id) : id{ _id }
    {
        this->init();
    }

    Keyframe(T _id, double _timestamp) : id{ _id }, timestamp{ _timestamp }
    {
        this->init();
    }

    void init()
    {
        this->pose.setIdentity();
        this->cloud.reset();
        this->normals.clear();
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

    bool addFrame(const Keyframe<int>& frame);
    void removeFrame(const unsigned int& idx);

    KeyframeMap<int> frames() const
    {
        return this->frames_;
    }

    KeyframeMap<int> operator()() const
    {
        return this->frames_;
    }

    void setCvHullAlpha(double alpha)
    {
        this->concave_hull_.setAlpha(alpha);
    }

    bool submapHasChanged() const
    {
        return this->submap_has_changed_;
    }

    PointCloud::Ptr submapCloud() const
    {
        return this->submap_cloud_;
    }

    FrameNormals submapNormals() const
    {
        return this->submap_normals_;
    }

    unsigned int size()
    {
        return this->frames_.size();
    }

    bool setFramePose(const unsigned int idx, const Eigen::Isometry3f& pose);

    Eigen::Isometry3f getFramePose(const unsigned int idx) const;
    PointCloud::Ptr getFrameCloud(const unsigned int idx) const;
    FrameNormals getFrameNormals(const unsigned int idx) const;

    void buildMap();
    void buildSubmap(Eigen::Vector3f pose);

    PointCloud::Ptr getMap();
    PointCloud::Ptr getMapDS(const double leaf_size);

private:
    void computeConvexHull();
    void computeConcaveHull();
    void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frame_indices);

    KeyframeMap<int> frames_;

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
