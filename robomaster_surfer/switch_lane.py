"""This is the controller for the RoboMaster Surfer project.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from enum import Enum
from multiprocessing import Manager

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .vision import Camera

SAVE_VIDEO = False
EPSILON = 0.01


class Lane:
    """Class used to describe the virtual lanes
    """

    def __init__(self, lane_id, name, pos):
        """initislize lane object

        :param lane_id: id of the lane
        :param name: name of the lane
        :param pos: y pos of the lane
        """

        self.id = lane_id
        self.name = name
        self.pos_y = pos


class State(Enum):
    """Enum used to easily understand the state of the RM
    """
    FORWARD = 0
    DECIDING = 1
    CHECKING = 2


class PDController:
    """PD controller class
    """

    def __init__(self, kp=2., kd=0.42):
        """Initialize the proportional controller

        :param kp: multiplicative constant used 
                   by the proportional controller
        :param kd: multiplicative constant used
                   by the derivative part of the controller
        """
        self.kp = kp
        self.kd = kd
        self.last_value = None

    def update_lat_vel(self, val, dt):
        i = 0
        if self.last_value is not None:
            i = ((val - self.last_value)/dt)*self.kd
        self.last_value = val

        return val*self.kp + i

    def update_ang_vel(self, des_theta, curr_theta):

        return self.kp * (des_theta - curr_theta)


class ControllerNode(Node):
    """
    Controller node to update RoboMaster position
    """

    def __init__(self):
        super().__init__('controller_node')

        self.timer = None
        self.lin_vel = 0.5  # desired linear vel to keep
        self.lat_vel = 0.0
        self.ang_vel = 0.0

        self.dt = 1/20
        self.check_every = 10
        self.camera_count = 0
        self.l_obs, self.r_obs = False, False

        self.num_lanes = 3
        self.corridor_size = 1.5

        self.lane_size = self.corridor_size / self.num_lanes
        self.lanes = self.create_lanes()

        self.cur_frame_id = None

        self.init_lane = self.lanes[1]
        self.current_lane = self.init_lane
        self.next_lane = None
        self.pc = PDController(kp=2.3)

        self.state = State(0)

        self.pose = None
        manager = Manager()
        self.last_decision = manager.Value("i", 1)

        self.camera = Camera(self, self.last_decision,
                             save_data=False, stream_data=True)

        self.init_theta = None
        self.theta = None
        self.switching = False

        # Create a publisher for the topic 'cmd_vel'
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 1)

        # Create a subscriber for RM Odometry
        self.pose_subscriber = self.create_subscription(
            Odometry, 'odom', self.pose_callback, 1)

    def patience_check(self, frame):
        """ Check if there are obstacles on the left
        and on the right of the RoboMaster

        :param frame: frame to use to perform check
        :return: tuple of booleans (l, r) True if the
                 left (right) lane contains an obstacle
        """
        tot = len(frame)

        counted_l = 0
        counted_r = 0
        # count number of obstacles pixels in the left and right of the image
        # checking 1 pixel every 5 to improve preformance
        for i in range(0, tot, 5):
            if frame[i][-1][2] > 125 and frame[i][-1][0] > 55:
                counted_r += 1
            if frame[i][0][2] > 125 and frame[i][0][0] > 55:
                counted_l += 1
        tot = tot/5
        left = counted_l/tot
        right = counted_r/tot

        self.get_logger().info(str(round(left, 3)))
        self.get_logger().info(str(round(right, 3)))
        return left > 0.58, right > 0.58

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
        The function takes in a message from the topic `/odom` and
        stores the position and orientation of the RoboMaster in the
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
        self.timer = self.create_timer(1 / 40, self.update_callback)

    def stop(self):
        """
        Set all velocities to zero
        """
        cmd_vel = Twist()
        self.vel_publisher.publish(cmd_vel)

    def sensed_obstacle_in_lane(self):
        """Returns true if an obstacle is sensed in the
        current lane, 0 otherwise
        """
        if self.last_decision.value != 1:
            return True
        return False

    def choose_next_lane(self):
        if self.last_decision.value != 1:
            self.get_logger().info(str(self.last_decision.value))

            if self.last_decision.value == 0:
                return self.lanes[(self.current_lane.id-1) % 3]
            elif self.last_decision.value == 2:
                return self.lanes[(self.current_lane.id+1) % 3]

        return self.current_lane

    def sensed_lat_obstacles(self, frame_id):
        """ returns the estimated presence of an obstacle
        on the left or on the right based on the patience
        check

        :param frame_id: frame id
        :return: True if an obstacle is estimated, false otherwise
        """
        tmp = False
        if self.cur_frame_id is None:
            if self.current_lane.id == 0:
                tmp = self.r_obs
            elif self.current_lane.id == 2:
                tmp = self.l_obs
            else:
                if self.next_lane.id == 0:
                    tmp = self.l_obs
                elif self.next_lane.id == 2:
                    tmp = self.r_obs
            if tmp:
                self.cur_frame_id = frame_id
        return tmp

    async def update_callback(self):
        """
        Update linear and angular velocities and publish them to ROS.
        """
        # Wait until an image is available
        if self.camera.frame is None or self.pose is None:
            return

        # self.get_logger().info(str(self.state))
        if (self.camera.frame_id - self.camera_count) >= self.check_every:
            self.camera_count = self.camera.frame_id
            self.l_obs, self.r_obs = self.patience_check(self.camera.frame)

        if self.state == State.FORWARD:
            if self.sensed_obstacle_in_lane():
                self.state = State.DECIDING
            else:
                err = self.current_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(err, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)

        if self.state == State.DECIDING:
            self.next_lane = self.choose_next_lane()
            if self.next_lane != self.current_lane:
                self.state = State.CHECKING

        if self.state == State.CHECKING:

            self.pc.last_value = None
            if self.switching or (self.cur_frame_id is None and not self.sensed_lat_obstacles(self.camera.frame_id)):
                diff = self.next_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(diff, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)
                self.switching = True

                if abs(diff) <= EPSILON:
                    self.get_logger().debug("Finished switching, now should go straight")
                    self.current_lane = self.next_lane
                    self.state = State.FORWARD
                    self.switching = False
                    self.get_logger().info("Forward")

            else:
                if self.camera.frame_id - self.cur_frame_id >= 30:
                    self.cur_frame_id = None
                    self.get_logger().info("restart switching lane operation")
                err = self.current_lane.pos_y - self.pose.y
                self.lat_vel = self.pc.update_lat_vel(err, self.dt)
                self.ang_vel = self.pc.update_ang_vel(
                    self.init_theta, self.theta)


        cmd_vel = Twist()
        cmd_vel.linear.x = self.lin_vel

        cmd_vel.linear.y = self.lat_vel
        cmd_vel.angular.z = self.ang_vel

        self.vel_publisher.publish(cmd_vel)

    def print_debug(self):
        """
        It prints the current lane, the destination lane, the lateral velocity, the linear velocity, and the
        switch period
        """
        self.get_logger().debug(
            f'Curr. lane: {self.current_lane}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Dest. vel: {self.next_lane}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Lat. vel: {self.lat_vel}', throttle_duration_sec=0.5)
        self.get_logger().debug(
            f'Lin. vel: {self.lin_vel}', throttle_duration_sec=0.5)


def main():
    # Initialize the ROS client library
    rclpy.init(args=sys.argv)

    # Create an instance of your node class
    node = ControllerNode()
    node.start()

    # Keep processing events until someone manually shuts down the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.camera.stop()
        pass

    # Ensure the RM is stopped before exiting
    node.stop()


if __name__ == '__main__':
    main()
