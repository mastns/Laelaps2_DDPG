#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Bool.h"
#include <laelaps_control/Toe.h>
#include <math.h>
#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>
#include <iostream>
//
#define PI2 1.5708
#define PI 3.14159265359
#define LINK1 0.25
#define LINK2 0.35
#define ARADIUS 0.06
#define BRADIUS 0.03
#define FREQ 30.0 // frequency
//
float init_toex = 0.0;
float init_toey = -0.58; 
// 
std::tuple<float, float> theta1;
std::tuple<float, float> theta2;
std::tuple<float, float> theta3;
std::tuple<float, float> theta4;
//
std_msgs::Float64 RH_msg_hip;
std_msgs::Float64 RH_msg_knee;
std_msgs::Float64 RF_msg_hip;
std_msgs::Float64 RF_msg_knee;
std_msgs::Float64 LF_msg_hip;
std_msgs::Float64 LF_msg_knee;
std_msgs::Float64 LH_msg_hip;
std_msgs::Float64 LH_msg_knee;
//
std_msgs::Bool RH_toe_nan;
std_msgs::Bool RF_toe_nan;
std_msgs::Bool LF_toe_nan;
std_msgs::Bool LH_toe_nan;
//
float p_x = 0.0;
float p_y = 0.0;
float secs = 0.0;
float nsecs = 0.0;
float gazebo_time = 0.0;
bool paused = false;
//

std::tuple<float, float> inverseKinematics(float &px, float &py)
{
    float cos_theta = 0.0;
    float sin_theta_plus = 0.0;
    float sin_theta_minus = 0.0;
    float theta_1 = 0.0;
    float theta_3 = 0.0;
    //
    cos_theta = ((pow(px, 2.0) + pow(py, 2.0) - pow(LINK1, 2.0) - pow(LINK2, 2.0)) / (2 * LINK1 * LINK2));
    sin_theta_plus = sqrt(1 - cos_theta * cos_theta);
    sin_theta_minus = -sqrt(1 - cos_theta * cos_theta);
    theta_1 = atan2(sin_theta_plus, cos_theta);
    theta_3 = atan2(py, px) - atan2(LINK2 * sin_theta_plus, LINK1 + LINK2 * cos_theta);
    //
    theta_3 = theta_3 + PI2; //radians
    theta_1 = theta_1 + theta_3;
    //
    return std::make_tuple(theta_1, theta_3);
}

void toe1inverseKinematics(float px, float py)
{
    theta1 = inverseKinematics(px, py);
    if ( !(std::isnan( std::get<1>(theta1) ) )  && !(std::isnan( std::get<0>(theta1) ) ) ){
        RH_msg_hip.data = std::get<1>(theta1);
        RH_msg_knee.data = std::get<0>(theta1);
        RH_toe_nan.data = false;
    }
    else
    {
        RH_toe_nan.data = true;
        std::cout << "is nan" <<'\n';
    }
    
}

void toe2inverseKinematics(float px, float py)
{
    theta2 = inverseKinematics(px, py);
    if ( !(std::isnan( std::get<1>(theta2) ) )  && !(std::isnan( std::get<0>(theta2) ) ) ){
        RF_msg_hip.data = std::get<1>(theta2);
        RF_msg_knee.data = std::get<0>(theta2);
        RF_toe_nan.data = false;
    }
    else
    {
        RF_toe_nan.data = true;
        std::cout << "is nan" <<'\n';
    }
}

void toe3inverseKinematics(float px, float py)
{
    theta3 = inverseKinematics(px, py);
    if ( !(std::isnan( std::get<1>(theta3) ) )  && !(std::isnan( std::get<0>(theta3) ) ) ){
        LF_msg_hip.data = std::get<1>(theta3);
        LF_msg_knee.data = std::get<0>(theta3);
        LF_toe_nan.data = false;
    }
    else
    {
        LF_toe_nan.data = true;
        std::cout << "is nan" <<'\n';
    }
}

void toe4inverseKinematics(float px, float py)
{
    theta4 = inverseKinematics(px, py);
    if ( !(std::isnan( std::get<1>(theta4) ) )  && !(std::isnan( std::get<0>(theta4) ) ) ){
        LH_msg_hip.data = std::get<1>(theta4);
        LH_msg_knee.data = std::get<0>(theta4);
        LH_toe_nan.data = false;
    }
    else
    {
        LH_toe_nan.data = true;
        std::cout << "is nan" <<'\n';
    }
}

std::tuple<float, float> ellipsoid(float x_center, float y_center, float df)
{
    gazebo_time = secs + (float((nsecs / 1000000000.0)));
    p_x = (x_center + BRADIUS * sin(-FREQ * gazebo_time - df));
    p_y = (y_center + ARADIUS * cos(-FREQ * gazebo_time - df));
    if (p_y < y_center)
    {
        p_y = y_center;
    };
    return std::make_tuple(p_x, p_y);
};

void toe1Callback(const laelaps_control::Toe msg)
{
    std::tuple<float, float> point;
    point = ellipsoid(msg.toex, msg.toey, msg.phase);
    toe1inverseKinematics(std::get<0>(point), std::get<1>(point));
}

void toe2Callback(const laelaps_control::Toe msg)
{
    std::tuple<float, float> point;
    point = ellipsoid(msg.toex, msg.toey, msg.phase);
    toe2inverseKinematics(std::get<0>(point), std::get<1>(point));
}

void toe3Callback(const laelaps_control::Toe msg)
{
    std::tuple<float, float> point;
    point = ellipsoid(msg.toex, msg.toey, msg.phase);
    toe3inverseKinematics(std::get<0>(point), std::get<1>(point));
}

void toe4Callback(const laelaps_control::Toe msg)
{
    std::tuple<float, float> point;
    point = ellipsoid(msg.toex, msg.toey, msg.phase);
    toe4inverseKinematics(std::get<0>(point), std::get<1>(point));
}

void time_cb(ConstWrenchStampedPtr &_msg)
{
    secs = _msg->time().sec();
    nsecs = _msg->time().nsec();
}

int main(int _argc, char **_argv)
{
    ros::init(_argc, _argv, "toe4_pos_publisher");
    ROS_INFO("Hello World! // toe4_ellipse_publisher // ");
    ros::NodeHandle node_ros;
    ros::Rate loop_rate(1000);
    gazebo::client::setup(_argc, _argv);
    gazebo::transport::NodePtr node_gazebo(new gazebo::transport::Node());
    node_gazebo->Init();
    //
    gazebo::transport::SubscriberPtr time_sub = node_gazebo->Subscribe("~/laelaps/LF_joint_hip/LF_force_torque/wrench", time_cb);
    //
    ros::Publisher LF_hip_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/LF_hip/command", 1);
    ros::Publisher LF_knee_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/LF_knee/command", 1);
    ros::Publisher RF_hip_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/RF_hip/command", 1);
    ros::Publisher RF_knee_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/RF_knee/command", 1);
    ros::Publisher LH_hip_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/LH_hip/command", 1);
    ros::Publisher LH_knee_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/LH_knee/command", 1);
    ros::Publisher RH_hip_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/RH_hip/command", 1);
    ros::Publisher RH_knee_pub = node_ros.advertise<std_msgs::Float64>("/laelaps_robot/RH_knee/command", 1);
    //
    ros::Publisher RH_nan_pub = node_ros.advertise<std_msgs::Bool>("/laelaps_robot/RH_toe/nan", 1);
    ros::Publisher RF_nan_pub = node_ros.advertise<std_msgs::Bool>("/laelaps_robot/RF_toe/nan", 1);
    ros::Publisher LF_nan_pub = node_ros.advertise<std_msgs::Bool>("/laelaps_robot/LF_toe/nan", 1);
    ros::Publisher LH_nan_pub = node_ros.advertise<std_msgs::Bool>("/laelaps_robot/LH_toe/nan", 1);
    //
    ros::Subscriber toe1_sub = node_ros.subscribe("/laelaps_robot/toe1/command", 1, toe1Callback);
    ros::Subscriber toe2_sub = node_ros.subscribe("/laelaps_robot/toe2/command", 1, toe2Callback);
    ros::Subscriber toe3_sub = node_ros.subscribe("/laelaps_robot/toe3/command", 1, toe3Callback);
    ros::Subscriber toe4_sub = node_ros.subscribe("/laelaps_robot/toe4/command", 1, toe4Callback);
    //
    toe1inverseKinematics(init_toex, init_toey);
    toe2inverseKinematics(init_toex, init_toey);
    toe3inverseKinematics(init_toex, init_toey);
    toe4inverseKinematics(init_toex, init_toey);
    //
    for (int i = 0; i <= 25; i++)
    {
        //
        std::cout << "publishing initial angles" <<'\n';
        //
        RH_hip_pub.publish(RH_msg_hip);
        RH_knee_pub.publish(RH_msg_knee);
        //
        RF_hip_pub.publish(RF_msg_hip);
        RF_knee_pub.publish(RF_msg_knee);
        //
        LF_hip_pub.publish(LF_msg_hip);
        LF_knee_pub.publish(LF_msg_knee);
        //
        LH_hip_pub.publish(LH_msg_hip);
        LH_knee_pub.publish(LH_msg_knee);
        //
        ros::spinOnce();
        loop_rate.sleep();
    }

    while (ros::ok())
    {
        //if ( !RH_toe_nan.data && !RF_toe_nan.data && !LF_toe_nan.data && !LH_toe_nan.data ){   
            RH_hip_pub.publish(RH_msg_hip);
            RH_knee_pub.publish(RH_msg_knee);
	        //
            RF_hip_pub.publish(RF_msg_hip);
            RF_knee_pub.publish(RF_msg_knee);
	        //
            LF_hip_pub.publish(LF_msg_hip);
            LF_knee_pub.publish(LF_msg_knee);
	        //
            LH_hip_pub.publish(LH_msg_hip);
            LH_knee_pub.publish(LH_msg_knee);
        //}
        RH_nan_pub.publish(RH_toe_nan);
        RF_nan_pub.publish(RF_toe_nan);
        LF_nan_pub.publish(LF_toe_nan);
        LH_nan_pub.publish(LH_toe_nan);
        //
        ros::spinOnce();
        gazebo::common::Time::MSleep(1);
        loop_rate.sleep();
    }
    // Make sure to shut everything down.
    gazebo::client::shutdown();
    return 0;
}
