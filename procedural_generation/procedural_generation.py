"""This is the launch script for the whole environment, the procedural
generation and RoboMaster controller node.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess

try:
    import sim
except:
    pass
import argparse
import os
import time

from mazeGenerator import generate_lab, state_from_row


def place_object(clientID, handle, position):
    """Place a new object of the same type of the given handle
    in the given position.

    :param clientID: ID of the CoppeliaSim client
    :param handle: id of the object which you want to duplicate
    :param position: position where the new object needs to be placed in
    :returns: handle of the new object
    """
    _, new_object_handle = sim.simxCopyPasteObjects(
        clientID, [handle], sim.simx_opmode_blocking)
    new_object_handle = new_object_handle[0]
    # Make new object not collidable
    sim.simxSetObjectIntParameter(
        clientID, new_object_handle, 3004, 0, sim.simx_opmode_blocking)
    # Move new object to the desidered position
    sim.simxSetObjectPosition(
        clientID, new_object_handle, -1, position, sim.simx_opmode_oneshot)
    # Make the object collidable
    sim.simxSetObjectIntParameter(
        clientID, new_object_handle, 3004, 1, sim.simx_opmode_oneshot)

    return new_object_handle


def remove_obstacle(clientID, handle):
    """Remove from the scene the object with the given handle.

    :param clientID: ID of the CoppeliaSim client
    :param handle: id of the object to remove
    """
    sim.simxRemoveObject(
        clientID, handle, sim.simx_opmode_oneshot)


def move_object(clientID, handle, position):
    """Move an object in a new position.

    :param clientID: ID of the CoppeliaSim client
    :param handle: id of the object to move
    :param position: new position of the object
    """
    # Make object non collidable
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 0, sim.simx_opmode_blocking)
    # Move oject to the desidered position
    sim.simxSetObjectPosition(
        clientID, handle, -1, position, sim.simx_opmode_oneshot)
    # Make the object collidable
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 1, sim.simx_opmode_oneshot)


def place_objects(symbolic_map, handlers,
                  cur_distance, offset_world,
                  offset_blocks, object_size):
    """place the specified objects in the symbolic map inside the
    simulation environment at a computed position.
    Used to generate chunks.

    :param symbolic_map: matrix representing the symbolic map
                         of the objects position in the environment
    :param handlers: list of handlers for the objects in the symbolic map
    :param cur_distance: distance of the current chunk from the first one in
                         the simulation
    :param offset_world: offset of the first chunk from the
                         origin in term of x axis
    :param offset_blocks: offset of the blocks from the computed position
    :param object_size: size of the objects to place
    :returns: a collection of the objects in the generated chunk
    """
    objects_collection = {1: [], 2: [], 3: [], 4: []}
    for i in range(len(symbolic_map)):
        for j in range(len(symbolic_map[i])):
            # If the value in the symbolic map is 0 no object is to be placed
            if symbolic_map[i][j] != 0:
                # retrieve handler of the specified type of object
                handle = handlers[symbolic_map[i][j] - 1]
                # compute position of the object
                position = (cur_distance + offset_world - offset_blocks + i *
                            object_size, j * object_size + offset_blocks, 0.2)
                # If the object is not an obstacle, its position will
                # be closer to the floor
                if symbolic_map[i][j] != 1:
                    position = (cur_distance + offset_world - offset_blocks + i *
                                object_size, j * object_size + offset_blocks, -0.04)

                # create the new object and add it to the collection
                objects_collection[symbolic_map[i][j]].append(
                    place_object(clientID, handle, position))
                time.sleep(0.05)

    return objects_collection


def shell_source(script):
    """Source the project"""
    pipe = subprocess.Popen(". %s && env -0" % script,
                            stdout=subprocess.PIPE,
                            shell=True, executable='/bin/bash')
    output = pipe.communicate()[0].decode('utf-8')
    output = output[:-1]
    env = {}
    for line in output.split('\x00'):
        line = line.split('=', 1)
        env[line[0]] = line[1]
    os.environ.update(env)


def main(clientID):

    if clientID != -1:
        # Get Handlers of important objects
        _, hanlder_RM = sim.simxGetObjectHandle(
            clientID, "/RoboMaster", sim.simx_opmode_blocking)

        # _, handler_camera = sim.simxGetObjectHandle(
        #    clientID, "/DefaultCamera", sim.simx_opmode_blocking)

        _, hanlder_obstacle = sim.simxGetObjectHandle(
            clientID, "/Obstacle", sim.simx_opmode_blocking)

        _, handler_coin1 = sim.simxGetObjectHandle(
            clientID, "/Coin1", sim.simx_opmode_blocking)

        _, handler_coin2 = sim.simxGetObjectHandle(
            clientID, "/Coin2", sim.simx_opmode_blocking)

        _, handler_coin5 = sim.simxGetObjectHandle(
            clientID, "/Coin5", sim.simx_opmode_blocking)

        handler_blocks = [hanlder_obstacle,
                          handler_coin1, handler_coin2, handler_coin5]

        _, hanlder_wall_right = sim.simxGetObjectHandle(
            clientID, "/WallRight", sim.simx_opmode_blocking)

        _, hanlder_wall_left = sim.simxGetObjectHandle(
            clientID, "/WallLeft", sim.simx_opmode_blocking)

        _, pos_RM = sim.simxGetObjectPosition(
            clientID, hanlder_RM, -1, sim.simx_opmode_blocking)

        chunks_floor_handlers_collection = []
        cur_chunk_from_start = 0  # (number of created chunks * length tile)

        corridor_size = 1.5
        obstacle_size = corridor_size / 3
        offset_blocks = obstacle_size / 2

        # For each floor of the initial chunks
        for i in range(5):
            # Save chunks handlers
            _, id = sim.simxGetObjectHandle(
                clientID, "/Tile[" + str(i) + "]", sim.simx_opmode_blocking)
            chunks_floor_handlers_collection.append(id)

        # This position is used to choose when to spawn a new chunk and
        # delete the old one
        cur_chunk_x = pos_RM[0]

        # Compute position of the new chunk to create
        _, pos_chunk = sim.simxGetObjectPosition(
            clientID, chunks_floor_handlers_collection[0],
            -1, sim.simx_opmode_blocking)
        pos_chunk = (pos_chunk[0] + 20, pos_chunk[1], pos_chunk[2])

        # Get left wall position
        _, pos_wall = sim.simxGetObjectPosition(
            clientID, hanlder_wall_left, -1, sim.simx_opmode_blocking)

        # Compute offset of the x axis of the generated world w.r.t the origin
        offset_world = round(pos_RM[0])

        def update_chunks(tile_handler, handler_wall_left,
                          handler_wall_right, pos_tile, wall_pose):
            """ Reuse assets floor and walls of the old chunk to  
            create the new one

            :param tile_handler: handler of the floor of the old chunk
            :param handler_wall_left: handler of the left wall
            :param handler_wall_right: handler of the right wall
            :param pos_tile: position of the tile to place
            :param wall_pose: position of the left wall
            :returns: tuple with the new floor position and left wall position
            """
            # Set walls new position
            wlp = (wall_pose[0] + 5, wall_pose[1], wall_pose[2])
            wrp = (wall_pose[0] + 5, wall_pose[1] + 1.5, wall_pose[2])
            tp = (pos_tile[0] + 5, pos_tile[1], pos_tile[2])
            move_object(clientID, handler_wall_left, wlp)
            time.sleep(0.03)
            move_object(clientID, handler_wall_right, wrp)
            time.sleep(0.03)
            move_object(clientID, tile_handler, tp)
            time.sleep(0.03)

            return tp, wlp

        def symbolic_map_to_pos_list(symbolic_map, cur_chunk_dist):
            """Convert the symbolic map to a list of positions

            :param symbolic_map:
            :param cur_chunk_dist:
            :returns: positions of the objects in the symbolic_map
            """
            objects_positions = {1: [], 2: [], 3: [], 4: []}
            for i in range(len(symbolic_map)):
                for j in range(len(symbolic_map[0])):
                    if symbolic_map[i][j] != 0:
                        if symbolic_map[i][j] == 1:
                            objects_positions[symbolic_map[i][j]].append(
                                (cur_chunk_dist + offset_world - offset_blocks + i *
                                 obstacle_size, j * obstacle_size + offset_blocks,
                                 0.2))
                        else:
                            objects_positions[symbolic_map[i][j]].append(
                                (cur_chunk_dist + offset_world - offset_blocks + i *
                                 obstacle_size, j * obstacle_size + offset_blocks,
                                 -0.04))
            return objects_positions

        # START PROCEDURAL CODE

        # Generate first 5 chunks obstacles
        chunks_objects = []

        symbolic_map = generate_lab(0, 10, 3)
        chunks_objects.append(place_objects(symbolic_map, handler_blocks,
                                            cur_chunk_from_start, offset_world,
                                            offset_blocks, obstacle_size))
        cur_chunk_from_start += 5
        for i in range(4):
            symbolic_map = generate_lab(
                state_from_row(symbolic_map[-1]), 10, 1)
            chunks_objects.append(place_objects(symbolic_map, handler_blocks,
                                                cur_chunk_from_start, offset_world,
                                                offset_blocks, obstacle_size))
            cur_chunk_from_start += 5

        # Start procedural generation sequence
        generated_next_chunk = False
        start = False
        new_tile_obs = {1: [], 2: [], 3: [], 4: []}
        pos_obstacles = {1: [], 2: [], 3: [], 4: []}
        while True:
            # if the symbolic map of the next chunk to generate is not
            # yet created
            if not generated_next_chunk:

                # Generate symbolic map
                symbolic_map = generate_lab(
                    state_from_row(symbolic_map[-1]), 10, 1)

                # Retrieve position of objects to place in the chunk
                pos_obstacles = symbolic_map_to_pos_list(
                    symbolic_map,
                    cur_chunk_from_start)

                # Update distance from first chunk
                cur_chunk_from_start += 5
                generated_next_chunk = True

            _, pos_RM = sim.simxGetObjectPosition(
                clientID, hanlder_RM, -1, sim.simx_opmode_blocking)

            # If the RoboMaster is over the current chunk (starting a new one)
            if not start and abs(cur_chunk_x - pos_RM[0]) > 5:
                start = True
                moved_tile = chunks_floor_handlers_collection[0]
                pos_chunk, pos_wall = update_chunks(
                    moved_tile, hanlder_wall_left,
                    hanlder_wall_right, pos_chunk, pos_wall)
                chunks_floor_handlers_collection = chunks_floor_handlers_collection[1:]
                chunks_floor_handlers_collection.append(moved_tile)
                cur_chunk_x = pos_RM[0]

            # (to make code more efficient compute only
            # the pos of new obstacles), otherwise you should put 5
            # instead of 2 as range
            for el in range(1, 2):

                # If it is possible to reuse old assets
                if start and len(pos_obstacles[el]) > 0 and len(chunks_objects[0][el]) > 0:
                    # reuse assets
                    obstacle_pos = pos_obstacles[el][0]
                    pos_obstacles[el] = pos_obstacles[el][1:]
                    handle_to_move = chunks_objects[0][el][0]
                    chunks_objects[0][el] = chunks_objects[0][el][1:]
                    move_object(clientID, handle_to_move, obstacle_pos)
                    new_tile_obs[el].append(handle_to_move)

                elif start and len(pos_obstacles[el]) > 0:
                    # if no old asset are avalaible, create new ones
                    obstacle_pos = pos_obstacles[el][0]
                    pos_obstacles[el] = pos_obstacles[el][1:]
                    new_tile_obs[el].append(place_object(
                        clientID, handler_blocks[el - 1], obstacle_pos))

                elif start and len(chunks_objects[0][el]) > 0:
                    # if there are old assets to remove
                    handle_to_move = chunks_objects[0][el][0]
                    chunks_objects[0][el] = chunks_objects[0][el][1:]
                    remove_obstacle(clientID, handle_to_move)

                time.sleep(0.08)

            # If no more actions are required to update chunks
            if len(pos_obstacles[1]) == 0 and len(chunks_objects[0][1]) == 0:
                start = False
                generated_next_chunk = False
                chunks_objects = chunks_objects[1:] + [new_tile_obs]
                new_tile_obs = {1: [], 2: [], 3: [], 4: []}
    else:
        print('Failed connecting to remote API server')
    print('Program ended')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--ip', type=str, default='127.0.0.1')
    ip = argparser.parse_args().ip

    try:
        print('Program started')
        sim.simxFinish(-1)  # just in case, close all opened connections
        clientID = sim.simxStart(ip, 19997, True, True,
                                 5000, 5)  # Connect to CoppeliaSim
        # Load robomaster scene
        sim.simxLoadScene(clientID,
                          "/home/usi/dev_ws/src/robomaster_surfer/scenes/rm_surfer.ttt",
                          0,
                          sim.simx_opmode_blocking)

        # Start simulation
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        shell_source('/home/usi/dev_ws/install/setup.bash')

        # Start the robomaster ros bridge
        bridge = subprocess.Popen(
            "ros2 launch robomaster_ros main.launch model:=s1 name:=RoboMaster serial_number:=RM0001 video_resolution:=720",
            shell=True, executable='/bin/bash')
        time.sleep(1)
        # Start the roboaster controller
        controller = subprocess.Popen(
            "ros2 launch robomaster_surfer lane_switcher.launch.py", shell=True, executable='/bin/bash')
    except:
        bridge.terminate()
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        controller.terminate()

    try:
        main(clientID)
    except KeyboardInterrupt:
        print("Interrupted!")
        # Stop simulation
        bridge.terminate()
        controller.terminate()
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
