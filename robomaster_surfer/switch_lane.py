"""This is the controller for Task 1 of the second Robotics assignment.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import pi

from enum import Enum

import rclpy
from geometry_msgs.msg import Twist, Pose
from rclpy.node import Node
import random

class CorridorLane(Enum):
    LEFT = -1
    CENTER = 0
    RIGHT = 1

class SwitchState(Enum):
    STRAIGHT = 0
    SWITCHING = 1

class ControllerNode(Node):
    """
    Open-loop controller class to follow an 8 trajectory.
    The idea is to write an eight as two circumferences.
    Hence the same angular velocity is used but with different
    sign, alternating between the two.
    """

    def __init__(self):
        super().__init__('controller_node')

  
        self.lin_vel = 0.5  # desired linear vel to keep
        self.lat_vel = 0.0

        self.num_lanes = 3
        self.corridor_size = 1.5

        self.lane_size = self.corridor_size/self.num_lanes
        
        self.switch_period = 1

        self.current_lane = CorridorLane(0)
        self.next_lane = None
        self.crossing_num = 0

        self.state = SwitchState(0)
        self.prev_state = None

        self.timestamp = None

        self.pose = None

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 1)

        # Create a subscriber for RM Odometry
        self.pose_subscriber = self.create_subscription(Pose, 'odom', self.pose_callback, 3)

    def pose_callback(self, msg):
        self.pose = msg

    def start(self):
        """Create and immediately start a timer that will regularly publish commands
        """
        self.timer = self.create_timer(1/60, self.update_callback)

    def stop(self):
        """Set all velocities to zero
        """
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def update_callback(self):
        """Update linear and angular velocities and publish them to ROS.
        """
        cmd_vel = Twist()

        cmd_vel.linear.x = self.lin_vel
        
        if self.state == SwitchState.STRAIGHT:
                self.lat_vel = 0.0
                if random.uniform(0,1) > 0.995:
                    self.next_lane = self.switch_lane_rand()
                    self.get_logger().info(f'Next lane: {self.next_lane}')

                    if self.next_lane != self.current_lane:
                        
                        diff = (self.next_lane.value - self.current_lane.value)
                        sign = -1 if diff >= 0 else 1

                        crossing_num = abs(diff)
                        
                        self.switch_period = crossing_num

                        self.state = SwitchState.SWITCHING
                        self.timestamp = self.get_clock().now().nanoseconds
                        self.lat_vel = sign * self.lane_size


                    else:
                        pass
                        #self.get_logger().info('Next lane equals current lane, should go straight')
        else:
            dt = (self.get_clock().now().nanoseconds - self.timestamp) * 1e-9

            if dt >= self.switch_period:
                #self.get_logger().info("Finished switching, now should go straight")
                self.state = SwitchState.STRAIGHT
                self.current_lane = self.next_lane
                self.lat_vel = 0.0
                self.switch_period = None
                
                
        self.lin_vel = 0.5  # desired linear vel to keep
        self.lat_vel = 0.0

        cmd_vel.linear.y = self.lat_vel

        self.vel_publisher.publish(cmd_vel)
                       
    def switch_lane_rand(self):
        """Select random lane
        """
        return CorridorLane(random.choice([-1, 0, 1]))

    def print_debug(self):
        self.get_logger().info(f'Curr. lane: {self.current_lane}')
        self.get_logger().info(f'Dest. vel: {self.next_lane}')
        self.get_logger().info(f'Lat. vel: {self.lat_vel}')
        self.get_logger().info(f'Lin. vel: {self.lin_vel}')
        self.get_logger().info(f'Switch period: {self.switch_period}')



def main():
    # Initialize the ROS client library
    rclpy.init(args=sys.argv)

    # Create an instance of your node class
    node = ControllerNode()
    node.start()

    # Keep processings events until someone manually shuts down the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Ensure the Thymio is stopped before exiting
    node.stop()


if __name__ == '__main__':
    main()
