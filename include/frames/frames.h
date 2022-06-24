#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>

namespace dlo
{
using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;
using FrameNormals = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;

struct Keyframe
{
    Keyframe() : id{ -1 }
    {
    }

    int id;
    double timestamp;
    Eigen::Isometry3f pose;
    PointCloud::Ptr cloud;
    FrameNormals normals;
};

class Frames
{
public:
    struct Params
    {
        int submap_knn_;
        int submap_kcv_;
        int submap_kcc_;
    };

    Frames(const Params& params);
    ~Frames();

    bool addFrame(const Keyframe& frame);
    void removeFrame(const unsigned int& idx);

    std::map<int, Keyframe> frames() const
    {
        return frames_;
    }

    void setCvHullAlpha(double alpha)
    {
        this->concave_hull_.setAlpha(alpha);
    }

    bool submapHasChanged() const
    {
        return submap_has_changed_;
    }

    PointCloud::Ptr submapCloud() const
    {
        return submap_cloud_;
    }

    FrameNormals submapNormals() const
    {
        return submap_normals_;
    }

    unsigned int size()
    {
        return frames_.size();
    }

    bool setFramePose(const unsigned int idx, const Eigen::Isometry3f& pose);

    Eigen::Isometry3f getFramePose(const unsigned int idx) const;
    PointCloud::Ptr getFrameCloud(const unsigned int idx) const;
    FrameNormals getFrameNormals(const unsigned int idx) const;

    void buildSubmap(Eigen::Vector3f pose);

    void buildMap();
    PointCloud::Ptr getMap();
    PointCloud::Ptr getMapDS(const double leaf_size);

private:
    void computeConvexHull();
    void computeConcaveHull();
    void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frame_indices);

    std::map<int, Keyframe> frames_;

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

    Params params_;
};
}  // namespace dlo
