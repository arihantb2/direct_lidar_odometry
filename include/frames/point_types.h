#pragma once

#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

namespace dlo
{
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;                     // enforce SSE padding for correct memory alignment

typedef PointXYZIRPYT PoseType;
typedef pcl::PointXYZI PointType;

}  // namespace dlo

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT (dlo::PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))
// clang-format on
