/************************************************************
 *
 * Copyright (c) 2021, University of California, Los Angeles
 *
 * Authors: Kenny J. Chen, Brett T. Lopez
 * Contact: kennyjchen@ucla.edu, btlopez@ucla.edu
 *
 ***********************************************************/

#include "dlo/odom.h"

void controlC(int sig)
{
    dlo::OdomNode::abort();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dlo_odom_node");
    ros::NodeHandle nh("~");

    signal(SIGTERM, controlC);
    sleep(0.5);

    dlo::OdomNode node(nh);
    node.start();

    ros::spin();

    return 0;
}
