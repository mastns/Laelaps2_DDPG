/*
 * Copyright (C) 2012 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gazebo/gazebo_client.hh>
#include <msgs.hh>
#include <gazebo/transport/transport.hh>

#include <iostream>

gazebo::msgs::WorldStatistics my_sim_time;
gazebo::msgs::Time my_time;
//
bool run0 = false;
bool run1 = false;
bool run2 = false;
/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb(ConstWorldStatisticsPtr &_msg) {
  // my_time = _msg->DebugString();
  static float secs = 0;
  static int nsecs = 0;
  std::cout << "Paused: " << _msg->paused() << "\n";
  secs = _msg->sim_time().sec();
  nsecs = _msg->sim_time().nsec();

  if (!(_msg->paused())) {
    if ((secs >= 0) && (secs < 3) && (run0 == false)) {
      std::cout << "command0 \n";
      system("/home/tns/xacro_ws/src/xacro_model/laelaps_control/ros_control_commands0.sh");
      run0 = true;
    } else if ((secs >= 3) && (secs < 7) && (run1 == false)) {
      std::cout << "command1 \n";
      system("/home/tns/xacro_ws/src/xacro_model/laelaps_control/ros_control_commands1.sh");
      run1 = true;
    } else if ((secs >= 7) && (secs < 10) && (run2 == false)) {
      std::cout << "command2 \n";
      system("/home/tns/xacro_ws/src/xacro_model/laelaps_control/ros_control_commands2.sh");
      run2 = true;
    } else {
      std::cout << " Second: " <<  secs << "\n";
      std::cout << "nSecond: " << nsecs << "\n";
    }
    std::cout << " Second: " <<  secs << "\n";
    std::cout << "nSecond: " << nsecs << "\n";
    std::cout << "\n";
  }
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv) {
  // Load gazebo
  gazebo::client::setup(_argc, _argv);

  // Create our node for communication
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();

  // Listen to Gazebo world_stats topic
  gazebo::transport::SubscriberPtr sub = node->Subscribe("~/world_stats", cb);
  //gazebo::transport::SubscriberPtr sub2 = node->Subscribe("~/world_stats", cb2);

  // Busy wait loop...replace with your own code as needed.
  while (true) gazebo::common::Time::MSleep(10);

  // Make sure to shut everything down.
  gazebo::client::shutdown();
}
