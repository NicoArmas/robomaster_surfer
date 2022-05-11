"""This is the controller for Task 1 of the second Robotics assignment.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import sys
from math import pi

from enum import Enum

import rclpy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry

from rclpy.node import Node
from sensor_msgs.msg import Image
from .vision import Camera
import matplotlib.pyplot as plt
import cv2

SAVE_VIDEO = True
EPSILON = 1e-4


class Lane():

    def __init__(self, id, name, pos):

        self.id = id
        self.name = name
        self.pos_y = pos


class SwitchState(Enum):
    STRAIGHT = 0
    SWITCHING = 1


class ProportionalController():

    def __init__(self, kp=1):
        self.kp = kp

    def update_vel(self, des_y, curr_y):

        return self.kp * (des_y - curr_y)



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
        self.lin_vel = 0.5  # desired linear vel to keep
        self.lat_vel = 0.0

        self.num_lanes = 3
        self.corridor_size = 1.5

        self.lane_size = self.corridor_size / self.num_lanes
        self.lanes = self.create_lanes()

        self.switch_period = 1

        self.init_lane = self.lanes[1]
        self.current_lane = self.init_lane
        self.next_lane = None
        self.crossing_num = 0
        self.pc = ProportionalController(kp=1)

        self.state = SwitchState(0)
        self.prev_state = None

        self.timestamp = None

        self.pose = None
        self.camera = Camera(self, 1000)
        self.camera_topic = self.create_subscription(
            Image, 'camera/image_raw', self.camera.camera_callback, 1)

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 1)

        # Create a subscriber for RM Odometry
        self.pose_subscriber = self.create_subscription(
            Odometry, 'odom', self.pose_callback, 3)
    def create_lanes(self):

        lanes = []
        pos = self.corridor_size - (self.lane_size/2)


        for i, name in zip(range(self.num_lanes), ['left', 'center', 'right']):
            lanes.append(Lane(i, name, pos))

            pos -= self.lane_size

        return lanes
    def pose_callback(self, msg):
        self.pose = msg.pose.pose.position
        self.pose.y += self.init_lane.pos_y
        # self.get_logger().info(f"pose: {self.pose}")

    def start(self):
        """Create and immediately start a timer that will regularly publish commands
        """
        self.timer = self.create_timer(1 / 60, self.update_callback)

    def stop(self):
        """Set all velocities to zero
        """
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def update_callback(self):
        """Update linear and angular velocities and publish them to ROS.
        """
        # Wait until an image is available
        if self.camera.frame is None or self.pose is None:
            return

        # Save the video when the framebuffer is full.
        if SAVE_VIDEO and self.camera.frame_idx == 0:
            w = cv2.VideoWriter(
                'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
            for frame in self.camera.framebuffer:
                w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            w.release()
        cmd_vel = Twist()

        cmd_vel.linear.x = self.lin_vel
        self.get_logger().info(str(self.state))
        if self.state == SwitchState.STRAIGHT:
            self.get_logger().info("Lane position: " + str(self.current_lane.pos_y))
            self.get_logger().info("RM position: " + str(self.pose.y))

            self.lat_vel = self.pc.update_vel(
                self.current_lane.pos_y, self.pose.y)
            self.get_logger().info("Calc vel: " + str(self.lat_vel))

            # if random.uniform(0,1) > 2:
            #     self.next_lane = self.switch_lane_rand()
            #     self.get_logger().info(f'Next lane: {self.next_lane}')

            #     if self.next_lane != self.current_lane:

            #         diff = (self.next_lane - self.current_lane.id)
            #         sign = -1 if diff >= 0 else 1

            #         crossing_num = abs(diff)

            #         self.switch_period = crossing_num

            #         self.state = SwitchState.SWITCHING
            #         self.timestamp = self.get_clock().now().nanoseconds
            #         self.lat_vel = sign * self.lane_size


                # else:
                #     pass
                    # self.get_logger().info('Next lane equals current lane, should go straight')
        else:
            self.lat_vel = self.pc.update_vel(
                self.next_lane.pos_y, self.pose.y)

            if (self.next_lane.pos_y - self.pose.y) <= EPSILON:
                # self.get_logger().info("Finished switching, now should go straight")
                self.state = SwitchState.STRAIGHT
                self.current_lane = self.next_lane
                self.lat_vel = 0.0
                self.switch_period = None


        self.lin_vel = 0.5  # desired linear vel to keep

        cmd_vel.linear.y = self.lat_vel

        self.vel_publisher.publish(cmd_vel)

    def switch_lane_rand(self):
        """Select random lane
        """
        return self.lanes[random.choice([0, 1, 2])]

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