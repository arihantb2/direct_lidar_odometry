#include <frames/frames.h>

namespace dlo
{
Frames::Frames(const SubmapParams& params) : params_{ params }
{
    init();
}

Frames::Frames(const SubmapParams& params, const KeyframeMap<int>& keyframes) : params_{ params }
{
    init();

    frames_ = keyframes;
}

void Frames::init()
{
    frames_.clear();

    this->submap_cloud_ = PointCloud::Ptr(new PointCloud);
    this->submap_has_changed_ = true;
    this->submap_kf_idx_prev_.clear();

    this->convex_hull_.setDimension(3);
    this->concave_hull_.setDimension(3);
    this->concave_hull_.setKeepInformation(true);

    submap_cloud_ = PointCloud::Ptr(new PointCloud);
    map_cloud_ = PointCloud::Ptr(new PointCloud);

    map_changed_ = true;
    submap_has_changed_ = true;
}

Frames::~Frames()
{
    submap_cloud_.reset();
}

bool Frames::addFrame(const Keyframe<int>& frame)
{
    if (frames_.find(frame.id) != frames_.end())
    {
        return false;
    }

    frames_.insert({ frame.id, frame });

    if (map_cloud_->empty())
    {
        map_cloud_ = PointCloud::Ptr(new PointCloud);
    }

    *map_cloud_ += *frame.getTransformedCloud();
    return true;
}

void Frames::removeFrame(const unsigned int& idx)
{
    if (frames_.find(idx) == frames_.end())
    {
        return;
    }

    map_changed_ = true;
    frames_.erase(idx);
}

bool Frames::setFramePose(const unsigned int idx, const Eigen::Isometry3f& pose)
{
    if (frames_.find(idx) == frames_.end())
    {
        return false;
    }

    if (frames_.at(idx).pose.isApprox(pose, 1e-6))
    {
        return false;
    }

    frames_.at(idx).pose = pose;
    map_changed_ = true;

    return true;
}

Eigen::Isometry3f Frames::getFramePose(const unsigned int idx) const
{
    if (frames_.find(idx) == frames_.end())
    {
        return {};
    }

    return frames_.at(idx).pose;
}

PointCloud::Ptr Frames::getFrameCloud(const unsigned int idx) const
{
    if (frames_.find(idx) == frames_.end())
    {
        return {};
    }

    return frames_.at(idx).cloud;
}

FrameNormals Frames::getFrameNormals(const unsigned int idx) const
{
    if (frames_.find(idx) == frames_.end())
    {
        return {};
    }

    return frames_.at(idx).normals;
}

PointCloud::Ptr Frames::getMap()
{
    if (map_changed_)
    {
        buildMap();
    }

    return map_cloud_;
}

PointCloud::Ptr Frames::getMapDS(const double leaf_size)
{
    if (map_changed_)
    {
        buildMap();
    }

    PointCloud::Ptr map_cloud_ds(new PointCloud);

    voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxelgrid.setInputCloud(map_cloud_);
    voxelgrid.filter(*map_cloud_ds);

    return map_cloud_ds;
}

void Frames::buildMap()
{
    map_cloud_.reset();
    map_cloud_ = PointCloud::Ptr(new PointCloud);

    for (auto kf_pair : frames_)
    {
        *map_cloud_ += *kf_pair.second.getTransformedCloud();
    }

    map_changed_ = false;
}

void Frames::buildSubmap(Eigen::Vector3f curr_pose)
{
    // clear vector of keyframe indices to use for submap
    this->submap_kf_idx_curr_.clear();

    //
    // TOP K NEAREST NEIGHBORS FROM ALL KEYFRAMES
    //

    // calculate distance between current pose and poses in keyframe set
    std::vector<float> ds;
    std::vector<int> keyframe_nn;
    int i = 0;

    for (unsigned int kf_idx = 0; kf_idx < this->frames_.size(); kf_idx++)
    {
        const Eigen::Vector3f& kf_pose = this->frames_.at(kf_idx).pose.translation();
        float d = (curr_pose - kf_pose).norm();
        ds.push_back(d);
        keyframe_nn.push_back(kf_idx);
    }

    // get indices for top K nearest neighbor keyframe poses
    this->pushSubmapIndices(ds, this->params_.submap_knn_, keyframe_nn);

    //
    // TOP K NEAREST NEIGHBORS FROM CONVEX HULL
    //

    // get convex hull indices
    this->computeConvexHull();

    // get distances for each keyframe on convex hull
    std::vector<float> convex_ds;
    for (const auto& c : this->keyframe_convex_)
    {
        convex_ds.push_back(ds[c]);
    }

    // get indicies for top kNN for convex hull
    this->pushSubmapIndices(convex_ds, this->params_.submap_kcv_, this->keyframe_convex_);

    //
    // TOP K NEAREST NEIGHBORS FROM CONCAVE HULL
    //

    // get concave hull indices
    this->computeConcaveHull();

    // get distances for each keyframe on concave hull
    std::vector<float> concave_ds;
    for (const auto& c : this->keyframe_concave_)
    {
        concave_ds.push_back(ds[c]);
    }

    // get indicies for top kNN for convex hull
    this->pushSubmapIndices(concave_ds, this->params_.submap_kcc_, this->keyframe_concave_);

    //
    // BUILD SUBMAP
    //

    // concatenate all submap clouds and normals
    std::sort(this->submap_kf_idx_curr_.begin(), this->submap_kf_idx_curr_.end());
    auto last = std::unique(this->submap_kf_idx_curr_.begin(), this->submap_kf_idx_curr_.end());
    this->submap_kf_idx_curr_.erase(last, this->submap_kf_idx_curr_.end());

    // sort current and previous submap kf list of indices
    std::sort(this->submap_kf_idx_curr_.begin(), this->submap_kf_idx_curr_.end());
    std::sort(this->submap_kf_idx_prev_.begin(), this->submap_kf_idx_prev_.end());

    // check if submap has changed from previous iteration
    if (this->submap_kf_idx_curr_ == this->submap_kf_idx_prev_)
    {
        this->submap_has_changed_ = false;
    }
    else
    {
        this->submap_has_changed_ = true;

        // reinitialize submap cloud, normals
        PointCloud::Ptr submap_cloud(new PointCloud);
        PointCloud::Ptr transformed_cloud(new PointCloud);
        this->submap_normals_.clear();

        for (auto k : this->submap_kf_idx_curr_)
        {
            transformed_cloud.reset(new PointCloud);
            pcl::transformPointCloud(*this->frames_.at(k).cloud, *transformed_cloud, this->frames_.at(k).pose.matrix());

            // create current submap cloud
            *submap_cloud += *transformed_cloud;

            // grab corresponding submap cloud's normals
            this->submap_normals_.insert(std::end(this->submap_normals_), std::begin(this->frames_.at(k).normals), std::end(this->frames_.at(k).normals));
        }

        this->submap_cloud_ = submap_cloud;
        this->submap_kf_idx_prev_ = this->submap_kf_idx_curr_;
    }
}

void Frames::computeConvexHull()
{
    // at least 4 keyframes for convex hull
    if (this->frames_.size() < 4)
    {
        return;
    }

    // create a pointcloud with points at keyframes
    PointCloud::Ptr cloud = PointCloud::Ptr(new PointCloud);

    for (unsigned int kf_idx = 0; kf_idx < this->frames_.size(); kf_idx++)
    {
        PointType pt;
        pt.x = this->frames_.at(kf_idx).pose.translation().x();
        pt.y = this->frames_.at(kf_idx).pose.translation().y();
        pt.z = this->frames_.at(kf_idx).pose.translation().z();
        cloud->push_back(pt);
    }

    // calculate the convex hull of the point cloud
    this->convex_hull_.setInputCloud(cloud);

    // get the indices of the keyframes on the convex hull
    PointCloud::Ptr convex_points = PointCloud::Ptr(new PointCloud);
    this->convex_hull_.reconstruct(*convex_points);

    pcl::PointIndices::Ptr convex_hull_point_idx = pcl::PointIndices::Ptr(new pcl::PointIndices);
    this->convex_hull_.getHullPointIndices(*convex_hull_point_idx);

    this->keyframe_convex_.clear();
    for (int i = 0; i < convex_hull_point_idx->indices.size(); ++i)
    {
        this->keyframe_convex_.push_back(convex_hull_point_idx->indices[i]);
    }
}

void Frames::computeConcaveHull()
{
    // at least 5 keyframes for concave hull
    if (this->frames_.size() < 5)
    {
        return;
    }

    // create a pointcloud with points at keyframes
    PointCloud::Ptr cloud = PointCloud::Ptr(new PointCloud);

    for (unsigned int kf_idx = 0; kf_idx < this->frames_.size(); kf_idx++)
    {
        PointType pt;
        pt.x = this->frames_.at(kf_idx).pose.translation().x();
        pt.y = this->frames_.at(kf_idx).pose.translation().y();
        pt.z = this->frames_.at(kf_idx).pose.translation().z();
        cloud->push_back(pt);
    }

    // calculate the concave hull of the point cloud
    this->concave_hull_.setInputCloud(cloud);

    // get the indices of the keyframes on the concave hull
    PointCloud::Ptr concave_points = PointCloud::Ptr(new PointCloud);
    this->concave_hull_.reconstruct(*concave_points);

    pcl::PointIndices::Ptr concave_hull_point_idx = pcl::PointIndices::Ptr(new pcl::PointIndices);
    this->concave_hull_.getHullPointIndices(*concave_hull_point_idx);

    this->keyframe_concave_.clear();
    for (int i = 0; i < concave_hull_point_idx->indices.size(); ++i)
    {
        this->keyframe_concave_.push_back(concave_hull_point_idx->indices[i]);
    }
}

void Frames::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frame_indices)
{
    if (dists.empty())
    {
        return;
    }

    // maintain max heap of at most k elements
    std::priority_queue<float> pq;

    for (auto d : dists)
    {
        if (pq.size() >= k && pq.top() > d)
        {
            pq.push(d);
            pq.pop();
        }
        else if (pq.size() < k)
        {
            pq.push(d);
        }
    }

    // get the kth smallest element, which should be at the top of the heap
    float kth_element = pq.top();

    // get all elements smaller or equal to the kth smallest element
    for (int i = 0; i < dists.size(); ++i)
    {
        if (dists[i] <= kth_element)
        {
            this->submap_kf_idx_curr_.push_back(frame_indices[i]);
        }
    }
}
}  // namespace dlo
