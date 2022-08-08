/************************************************************
 *
 * Copyright (c) 2021, University of California, Los Angeles
 *
 * Authors: Kenny J. Chen, Brett T. Lopez
 * Contact: kennyjchen@ucla.edu, btlopez@ucla.edu
 *
 ***********************************************************/

#include "dlo/odom.h"

std::unique_ptr<dlo::OdomNode> odom_node_ptr;

void controlC(int sig)
{
    dlo::OdomNode::abort();
}

bool resetCallback(er_nav_msgs::SetLocalizationState::Request& request, er_nav_msgs::SetLocalizationState::Response& response)
{
    if (!odom_node_ptr)
    {
        std::cout << "[dlo::OdomNode::resetCallback]: Odom Node not running... Cannot reset\n";
    }

    std::cout << "[dlo::OdomNode::resetCallback]: Requested new map [" << request.file_tag << "], will overwrite current map [" << odom_node_ptr->mapName()
              << "] with [" << odom_node_ptr->mapSize() << "] frames\n";

    // Stop odom node
    odom_node_ptr->stop();

    // Delete odom node and create new object
    odom_node_ptr.reset();
    odom_node_ptr = std::make_unique<dlo::OdomNode>(ros::NodeHandle("~"), false);

    if (!request.file_tag.empty())
    {
        // Load new map
        if (odom_node_ptr->loadMap(request.file_tag))
        {
            std::cout << "[dlo::OdomNode::resetCallback]: Loaded new map [" << request.file_tag << "]\n";
        }
        else
        {
            std::cout << "[dlo::OdomNode::resetCallback]: Could not load map [" << request.file_tag << "]. Reset map\n";
        }
    }

    // Start odom node
    odom_node_ptr->start();

    response.success = true;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dlo_odom_node");
    ros::NodeHandle nh("~");

    signal(SIGTERM, controlC);
    sleep(0.5);

    odom_node_ptr = std::make_unique<dlo::OdomNode>(nh, true);
    if (!odom_node_ptr->mapName().empty())
    {
        // Load map
        odom_node_ptr->loadMap(odom_node_ptr->mapName());
    }
    odom_node_ptr->start();

    ros::ServiceServer reset_odom_server = nh.advertiseService("/localization/reset", &resetCallback);

    ros::spin();

    return 0;
}
