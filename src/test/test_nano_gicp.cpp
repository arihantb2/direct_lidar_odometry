#include <iostream>
#include <string>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <nano_gicp/nano_gicp.hpp>

std::default_random_engine rand_engine;

Eigen::Isometry3f getRandomTransform(const double max_translation, const double max_rotation_deg)
{
    // Generate random transform
    Eigen::Isometry3f T;
    std::uniform_real_distribution<double> rand_trans_dist(-1.0 * max_translation, max_translation);
    std::uniform_real_distribution<double> rand_rot_dist(-1.0 * max_rotation_deg * M_PI / 180.0, max_rotation_deg * M_PI / 180.0);

    Eigen::Vector3f translation(rand_trans_dist(rand_engine), rand_trans_dist(rand_engine), rand_trans_dist(rand_engine));
    Eigen::Quaternionf rotation(Eigen::AngleAxisf(rand_rot_dist(rand_engine), Eigen::Vector3f::UnitX()) *
                                Eigen::AngleAxisf(rand_rot_dist(rand_engine), Eigen::Vector3f::UnitY()) *
                                Eigen::AngleAxisf(rand_rot_dist(rand_engine), Eigen::Vector3f::UnitZ()));

    T.setIdentity();
    T = T.translate(translation).rotate(rotation);

    return T;
}

void printTransform(const Eigen::Isometry3f& transform, const std::string& prefix_str = "")
{
    std::cout << prefix_str << ": Translation (m) (" << transform.translation().transpose() << "), Rotation (deg): ("
              << transform.rotation().eulerAngles(0, 1, 2).transpose() << ")\n";
}

int main(int argc, char** argv)
{
    using PointType = pcl::PointXYZI;

    // Generate random transform
    Eigen::Isometry3f transform = getRandomTransform(20.0 /* metres */, 5.0 /* degrees */);

    printTransform(transform, "Ground Truth Transform");

    // Generate random point cloud
    std::uniform_real_distribution<double> rand_coordinate_dist(0.0, 50.0);
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    const unsigned int num_points = 1000;
    cloud->resize(num_points);
    for (unsigned int i = 0; i < num_points; ++i)
    {
        cloud->points[i].x = rand_coordinate_dist(rand_engine);
        cloud->points[i].y = rand_coordinate_dist(rand_engine);
        cloud->points[i].z = rand_coordinate_dist(rand_engine);
    }

    // Transform point cloud with random noise
    pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
    transformed_cloud->resize(num_points);
    const double max_trans_error = 1e-2;
    const double max_rot_error_deg = 1e-1;
    PointType transformed_point;
    Eigen::Matrix4f noisy_transform = transform.matrix() * getRandomTransform(max_trans_error, max_rot_error_deg).matrix();
    for (unsigned int i = 0; i < num_points; ++i)
    {
        noisy_transform = transform.matrix() * getRandomTransform(max_trans_error, max_rot_error_deg).matrix();
        transformed_point = pcl::transformPoint(cloud->points[i], Eigen::Affine3f(noisy_transform));
        transformed_cloud->points[i] = transformed_point;
    }

    // Run GICP
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    gicp.registerInputSource(cloud);
    // gicp.setInputSource(cloud);
    gicp.setInputTarget(transformed_cloud);

    pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>);

    gicp.align(*aligned_cloud, transform.matrix());
    Eigen::Isometry3f estimated_transform = Eigen::Isometry3f(gicp.getFinalTransformation());

    printTransform(estimated_transform, "Estimated Transform");
    std::cout << "Within 1e-1 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-1) << "\n";
    std::cout << "Within 1e-2 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-2) << "\n";
    std::cout << "Within 1e-3 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-3) << "\n";
    std::cout << "Within 1e-4 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-4) << "\n";
    std::cout << "Within 1e-5 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-5) << "\n";
    std::cout << "Within 1e-6 bound: " << std::boolalpha << estimated_transform.isApprox(transform, 1e-6) << "\n";

    return 0;
}
