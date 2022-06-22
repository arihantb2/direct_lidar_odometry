#include <iostream>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

void printTransform(const Eigen::Isometry3f& transform, const std::string& prefix_str = "")
{
    std::cout << prefix_str << ": Translation (m) (" << transform.translation().transpose() << "), Rotation (deg): ("
              << transform.rotation().eulerAngles(0, 1, 2).transpose() << ")\n";
}

int main(int argc, char** argv)
{
    using PointType = pcl::PointXYZ;

    PointType point_in_sensor_frame(1.0, 0.0, 0.0);

    std::cout << "Point in sensor frame: " << point_in_sensor_frame << std::endl;

    Eigen::Isometry3f T_body_to_sensor;
    T_body_to_sensor.setIdentity();
    T_body_to_sensor.translate(Eigen::Vector3f(0, 0, 0.5));
    T_body_to_sensor.rotate(Eigen::AngleAxisf(M_PI / 2.0, Eigen::Vector3f::UnitZ()));

    printTransform(T_body_to_sensor, "Ground Truth Transform");

    PointType transformed_point = pcl::transformPoint(point_in_sensor_frame, Eigen::Affine3f(T_body_to_sensor.matrix()));

    std::cout << "Point in body frame: " << transformed_point << std::endl;

    return 0;
}
