"""This is the controller for Task 1 of the second Robotics assignment.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
import sys
from enum import Enum

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from multiprocessing import Manager

from .vision import Camera

SAVE_VIDEO = False
EPSILON = 0.01


class Lane:

    def __init__(self, id, name, pos):

        self.id = id
        self.name = name
        self.pos_y = pos


class State(Enum):
    FORWARD = 0
    DECIDING = 1
    CHECKING = 2


class ProportionalController:

    def __init__(self, kp=2, kd=0.4):
        self.kp = kp
        self.kd = kd
        self.last_value = None

    def update_lat_vel(self, val, dt):
        i = 0
        if self.last_value != None:
            i = ((val - self.last_value)/dt)*self.kd
        self.last_value = val

        return val*self.kp + i

    def update_ang_vel(self, des_theta, curr_theta):

        return self.kp * (des_theta - curr_theta)

# class ProportionalController:

#     def __init__(self, kp=2):
#         self.kp = kp

#     def update_lat_vel(self, des_y, curr_y):
#         return self.kp * (des_y - curr_y)

#     def update_ang_vel(self, des_theta, curr_theta):
#         return self.kp * (des_theta - curr_theta)


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
        self.ang_vel = 0.0

        self.dt = 1/20

        self.num_lanes = 3
        self.corridor_size = 1.5

        self.lane_size = self.corridor_size / self.num_lanes
        self.lanes = self.create_lanes()

        self.switch_period = 1

        self.cur_frame_id = None

        self.init_lane = self.lanes[1]
        self.current_lane = self.init_lane
        self.next_lane = None
        self.crossing_num = 0
        self.pc = ProportionalController(kp=2)

        self.state = State(0)

        self.timestamp = None

        self.pose = None
        manager = Manager()
        self.last_decision = manager.Value("i", 1)

        self.camera = Camera(self, 600, self.last_decision, save_data=False,
                             save_video=SAVE_VIDEO, stream_data=True)
        self.is_saving = False

        self.init_theta = None
        self.theta = None

        self.timestamp = self.get_clock().now().nanoseconds

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 1)

        # Create a subscriber for RM Odometry
        self.pose_subscriber = self.create_subscription(
            Odometry, 'odom', self.pose_callback, 3)

        # self.steering_sequence = [0, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 1]
        self.cur = 1

    def check_col(self, col, frame):
        tot = len(frame[col])
        counted = 0
        for i in range(720):
            if frame[col][i][2] > 140 and frame[col][i][1] > 110:
                counted += 1
        return counted/tot > 0.8

    def check_left(self, frame):
        return self.check_col(0, frame)

    def check_right(self, frame):
        return self.check_col(-1, frame)

    def create_lanes(self):
        """
        It creates a list of lanes, each with a unique id, name, and position
        :return: A list of lanes.
        """
        lanes = []
        pos = self.corridor_size - (self.lane_size/2)

        for i, name in enumerate(['left', 'center', 'right']):
            lanes.append(Lane(i, name, pos))
            pos -= self.lane_size

        return lanes

    def pose_callback(self, msg):
        """
        The function takes in a message from the topic `/odom` and stores the position and orientation of the car in the
        variables `self.pose` and `self.theta` respectively

        :param msg: the message that is received from the topic
        """
        self.pose = msg.pose.pose.position
        if self.init_theta is None:
            self.init_theta = msg.pose.pose.orientation.z
        self.theta = msg.pose.pose.orientation.z
        self.pose.y += self.init_lane.pos_y
        # self.get_logger().info(f"pose: {self.pose}")

    def start(self):
        """
        Create and immediately start a timer that will regularly publish commands
        """
        self.timer = self.create_timer(1 / 60, self.update_callback)

    def stop(self):
        """
        Set all velocities to zero
        """
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def sensed_front_obstacles(self):
        if self.last_decision.value != 1:
            return True
        return False

    def lane_to_reach(self):
        if self.last_decision.value != 1:
            self.get_logger().info(str(self.last_decision.value))

            if self.last_decision.value == 0:
                return self.lanes[self.current_lane.id-1]
            elif self.last_decision.value == 2:
                return self.lanes[self.current_lane.id+1]

        return self.current_lane

    def sensed_lat_obstacles(self):
        tmp = False
        if self.cur_frame_id is None:
            if self.current_lane.id == 0:
                tmp = self.check_right(self.camera.frame)
            elif self.current_lane.id == 2:
                tmp = self.check_left(self.camera.frame)
            else:
                if self.next_lane.id == 0:
                    tmp = self.check_left(self.camera.frame)
                elif self.next_lane.id == 2:
                    tmp = self.check_right(self.camera.frame)
            if tmp:
                self.cur_frame_id = self.camera.frame_id
        return tmp

    async def update_callback(self):
        """
        Update linear and angular velocities and publish them to ROS.
        """
        # Wait until an image is available
        if self.camera.frame is None or self.pose is None and self.camera.move_buffer.empty():
            return

        # self.get_logger().info(str(self.state))

        if self.state == State.FORWARD:
            if self.sensed_front_obstacles():
                self.state = State.DECIDING
            else:
                err = self.current_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(err, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)

        # self.get_logger().info(str(self.state))

        if self.state == State.DECIDING:
            self.next_lane = self.lane_to_reach()
            if self.next_lane != self.current_lane:
                self.state = State.CHECKING

        # self.get_logger().info(str(self.state))

        if self.state == State.CHECKING:
            self.pc.last_value = None
            if not self.sensed_lat_obstacles() and self.cur_frame_id is None:
                # self.next_lane = self.lanes[0]  # ricordarsi di togliere
                diff = self.next_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(diff, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)

                if abs(diff) <= EPSILON:
                    self.get_logger().debug("Finished switching, now should go straight")
                    self.current_lane = self.next_lane
                    self.state = State.FORWARD
                    self.get_logger().info("Forward")

            else:
                if self.camera.frame_id - self.cur_frame_id >= 15:
                    self.cur_frame_id = None
                    self.get_logger().info("VAI USI VAI")
                err = self.current_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(err, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)

        # self.get_logger().info(str(self.state))
        # self.get_logger().info(str(''))

        cmd_vel = Twist()
        cmd_vel.linear.x = self.lin_vel

        cmd_vel.linear.y = self.lat_vel
        cmd_vel.angular.z = self.ang_vel

        self.vel_publisher.publish(cmd_vel)

    def print_debug(self):
        """
        It prints the current lane, the destination lane, the lateral velocity, the linear velocity, and the switch period
        """
        self.get_logger().debug(
            f'Curr. lane: {self.current_lane}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Dest. vel: {self.next_lane}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Lat. vel: {self.lat_vel}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Lin. vel: {self.lin_vel}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Switch period: {self.switch_period}', throttle_duration_sec=0.5)


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
        node.camera.stop()
        pass

    # Ensure the Thymio is stopped before exiting
    node.stop()


if __name__ == '__main__':
    main()
