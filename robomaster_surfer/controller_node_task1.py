"""This is the controller for Task 1 of the second Robotics assignment.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import pi

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ControllerNode(Node):
    """
    Open-loop controller class to follow an 8 trajectory.
    The idea is to write an eight as two circumferences.
    Hence the same angular velocity is used but with different
    sign, alternating between the two.
    """

    def __init__(self):
        super().__init__('controller_node')

        self.timer = None
        self.sign = 1  # Sign to handle steering direction
        self.radius = 0.4  # desired radius of the two circles in the 8
        self.linear_vel = 0.1  # desired linear vel to keep
        self.angular_vel = self.linear_vel/self.radius  # compute angular vel

        # N.B. we needed to multiply the computed period for a constant
        # to actually get the desired result
        self.period = ((2*pi)/self.angular_vel)*1.275  # Compute period

        # Initialize timestamp to have a reference of the elapsed time
        self.timestamp = self.get_clock().now().nanoseconds

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 1)

    def start(self):
        """Create and immediately start a timer that will regularly publish commands
        """
        self.timer = self.create_timer(1/30, self.update_callback)

    def stop(self):
        """Set all velocities to zero
        """
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def update_callback(self):
        """Update linear and angular velocities and publish them to ROS.
        """

        # Compute time elapsed since the starting of the circumference
        dt = (self.get_clock().now().nanoseconds-self.timestamp) * 1e-9
        self.get_logger().info("DT " + str(round(dt, 2)))

        if dt > self.period:
            # If the period is passed, change sign to start
            # the other circumference
            self.sign = -1 * self.sign
            self.get_logger().info("CHANGED SIGN: " + str(self.sign))
            # And track again timestamp to allow to check the elapsed time
            # for the new circumference
            self.timestamp = self.get_clock().now().nanoseconds

        cmd_vel = Twist()
        cmd_vel.angular.z = self.sign * self.angular_vel  # [rad/s]
        cmd_vel.linear.x = self.linear_vel  # [m/s]

        # Publish the command
        self.vel_publisher.publish(cmd_vel)


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
