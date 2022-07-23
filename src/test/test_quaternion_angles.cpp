#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

float getHalfAngle(const Eigen::Quaternionf& q1, const Eigen::Quaternionf& q2)
{
    const Eigen::Quaternionf q2_minus_q1 = q1 * q2.conjugate();
    return std::acos(q2_minus_q1.w());
}

float getHalfAngle(const Eigen::Quaternionf& q)
{
    return std::acos(q.w());
}

int main()
{
    float half_angle = getHalfAngle(Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.0, 0.0, 0.0), Eigen::Vector3f(0.0, 1.0, 0.0)));
    std::cout << "Calculated Half Angle: " << half_angle << "\n";
    std::cout << "Actual Half Angle    : " << 0.5 * M_PI / 2.0 << "\n";

    half_angle = getHalfAngle(Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.0, 0.0, 0.0), Eigen::Vector3f(-1.0, 0.0, 0.0)));
    std::cout << "Calculated Half Angle: " << half_angle << "\n";
    std::cout << "Actual Half Angle    : " << 0.5 * M_PI << "\n";

    half_angle = getHalfAngle(Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.0, 0.0, 0.0), Eigen::Vector3f(0.0, -1.0, 0.0)));
    std::cout << "Calculated Half Angle: " << half_angle << "\n";
    std::cout << "Actual Half Angle    : " << -0.5 * M_PI / 2.0 << "\n";

    half_angle = getHalfAngle(Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.0, 0.0, 0.0), Eigen::Vector3f(0.0, 0.0, 0.0)));
    std::cout << "Calculated Half Angle: " << half_angle << "\n";

    return 0;
}
