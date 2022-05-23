import subprocess
import shlex

try:
    import sim
except:
    pass
import time
import os

from mazeGenerator import generate_lab, state_from_row
import argparse


def place_obstacle(clientID, handle, position):
    _, a = sim.simxCopyPasteObjects(
        clientID, [handle], sim.simx_opmode_blocking)
    a = a[0]
    sim.simxSetObjectIntParameter(
        clientID, a, 3004, 0, sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(
        clientID, a, -1, position, sim.simx_opmode_oneshot)
    sim.simxSetObjectIntParameter(
        clientID, a, 3004, 1, sim.simx_opmode_oneshot)
    return a


def remove_obstacle(clientID, handle):
    sim.simxRemoveObject(
        clientID, handle, sim.simx_opmode_oneshot)


def move_obstacle(clientID, handle, position):
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 0, sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(
        clientID, handle, -1, position, sim.simx_opmode_oneshot)
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 1, sim.simx_opmode_oneshot)
    # handle2 = place_obstacle(clientID, handle, position)
    # remove_obstacle(clientID, handle)
    return handle


def place_obstacles(obs, handlers, switcher_count, offset_world, offset_blocks, obstacle_size):
    tile_obstacles = {1: [], 2: [], 3: [], 4: []}
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if obs[i][j] != 0:
                handle = handlers[obs[i][j] - 1]
                position = (switcher_count + offset_world - offset_blocks + i *
                            obstacle_size, j * obstacle_size + offset_blocks, 0.2)
                if obs[i][j] != 1:
                    position = (switcher_count + offset_world - offset_blocks + i *
                                obstacle_size, j * obstacle_size + offset_blocks, -0.04)
                tile_obstacles[obs[i][j]].append(
                    place_obstacle(clientID, handle, position))
                time.sleep(0.05)

    return tile_obstacles


def shell_source(script):
    """
    Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it.
    """

    pipe = subprocess.Popen(". %s && env -0" % script, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    output = pipe.communicate()[0].decode('utf-8')
    output = output[:-1]  # fix for index out for range in 'env[ line[0] ] = line[1]'

    env = {}
    # split using null char
    for line in output.split('\x00'):
        line = line.split('=', 1)
        # print(line)
        env[line[0]] = line[1]

    os.environ.update(env)


def main(clientID):
    # Load robomaster scene
    sim.simxLoadScene(clientID,
                      "/home/usi/dev_ws/src/robomaster_surfer/scenes/rm_surfer.ttt",
                      0,
                      sim.simx_opmode_blocking)

    # Start simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

    shell_source('/home/usi/dev_ws/install/setup.bash')

    # Start the robomaster ros bridge
    subprocess.Popen(
        "ros2 launch robomaster_ros main.launch model:=s1 name:=RoboMaster serial_number:=RM0001 video_resolution:=720",
        shell=True, executable='/bin/bash')

    # Start the roboaster controller
    subprocess.Popen("ros2 launch robomaster_surfer lane_switcher.launch.py", shell=True, executable='/bin/bash')

    if clientID != -1:
        # Get Handlers of important objects
        _, hanlder_RM = sim.simxGetObjectHandle(
            clientID, "/RoboMaster", sim.simx_opmode_blocking)

        _, handler_camera = sim.simxGetObjectHandle(
            clientID, "/DefaultCamera", sim.simx_opmode_blocking)

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

        tiles_handlers = []
        switcher_count = 0  # (number of created tiles * length tile)

        corridor_size = 1.5
        obstacle_size = corridor_size / 3
        offset_blocks = obstacle_size / 2

        for i in range(5):
            # Save tiles handlers
            _, id = sim.simxGetObjectHandle(
                clientID, "/Tile[" + str(i) + "]", sim.simx_opmode_blocking)
            tiles_handlers.append(id)
        cur_tile_x = pos_RM[0]

        _, pos_tile = sim.simxGetObjectPosition(
            clientID, tiles_handlers[0], -1, sim.simx_opmode_blocking)
        pos_tile = (pos_tile[0] + 20, pos_tile[1], pos_tile[2])

        # Get walls position
        _, pos_wall = sim.simxGetObjectPosition(
            clientID, hanlder_wall_left, -1, sim.simx_opmode_blocking)

        offset_world = round(pos_RM[0])

        def update_tiles(tile_handler, hanlder_wall_left, hanlder_wall_right, pos_tile, wall_pose):
            # Set walls new position
            wlp = (wall_pose[0] + 5, wall_pose[1], wall_pose[2])
            wrp = (wall_pose[0] + 5, wall_pose[1] + 1.5, wall_pose[2])
            tp = (pos_tile[0] + 5, pos_tile[1], pos_tile[2])
            hanlder_wall_left = move_obstacle(clientID, hanlder_wall_left, wlp)
            time.sleep(0.02)
            hanlder_wall_right = move_obstacle(clientID, hanlder_wall_right, wrp)
            time.sleep(0.02)
            tile_handler = move_obstacle(clientID, tile_handler, tp)
            time.sleep(0.02)

            return tp, wlp, tile_handler, hanlder_wall_left, hanlder_wall_right

        def from_map_to_pos_list(obs, switcher_count):
            obstacle_positions = {1: [], 2: [], 3: [], 4: []}
            for i in range(len(obs)):
                for j in range(len(obs[0])):
                    if obs[i][j] != 0:
                        if obs[i][j] == 1:
                            obstacle_positions[obs[i][j]].append((switcher_count + offset_world - offset_blocks + i *
                                                                  obstacle_size, j * obstacle_size + offset_blocks,
                                                                  0.2))
                        else:
                            obstacle_positions[obs[i][j]].append((switcher_count + offset_world - offset_blocks + i *
                                                                  obstacle_size, j * obstacle_size + offset_blocks,
                                                                  -0.04))
            return obstacle_positions

        # START PROCEDURAL CODE

        # Generate first 5 tiles obstacles

        tiles_objects = []
        for i in range(5):
            obs = generate_lab(0, 10, 3)
            tiles_objects.append(place_obstacles(obs, handler_blocks, switcher_count, offset_world,
                                                 offset_blocks, obstacle_size))
            switcher_count += 5

        generated_next_chunk = False
        start = False
        new_tile_obs = {1: [], 2: [], 3: [], 4: []}
        pos_obstacles = {1: [], 2: [], 3: [], 4: []}
        while True:
            if not generated_next_chunk:
                obs = generate_lab(state_from_row(obs[-1]), 10, 0)
                pos_obstacles = from_map_to_pos_list(
                    obs,
                    switcher_count)

                switcher_count += 5
                generated_next_chunk = True

            _, pos_RM = sim.simxGetObjectPosition(
                clientID, hanlder_RM, -1, sim.simx_opmode_blocking)

            if not start and abs(cur_tile_x - pos_RM[0]) > 5:
                start = True
                moved_tile = tiles_handlers[0]
                pos_tile, pos_wall, moved_tile, hanlder_wall_left, hanlder_wall_right = update_tiles(
                    moved_tile, hanlder_wall_left, hanlder_wall_right, pos_tile, pos_wall)
                tiles_handlers = tiles_handlers[1:]
                tiles_handlers.append(moved_tile)
                cur_tile_x = pos_RM[0]

            for el in range(1, 5):
                if start and len(pos_obstacles[el]) > 0 and len(tiles_objects[0][el]) > 0:
                    obstacle_pos = pos_obstacles[el][0]
                    pos_obstacles[el] = pos_obstacles[el][1:]
                    handle_to_move = tiles_objects[0][el][0]
                    tiles_objects[0][el] = tiles_objects[0][el][1:]
                    new_tile_obs[el].append(move_obstacle(
                        clientID, handle_to_move, obstacle_pos))

                elif start and len(pos_obstacles[el]) > 0:
                    obstacle_pos = pos_obstacles[el][0]
                    pos_obstacles[el] = pos_obstacles[el][1:]
                    new_tile_obs[el].append(place_obstacle(
                        clientID, handler_blocks[el - 1], obstacle_pos))

                elif start and len(tiles_objects[0][el]) > 0:
                    handle_to_move = tiles_objects[0][el][0]
                    tiles_objects[0][el] = tiles_objects[0][el][1:]
                    remove_obstacle(clientID, handle_to_move)

                time.sleep(0.05)

            if len(pos_obstacles[1]) == 0 and len(tiles_objects[0][1]) == 0:
                start = False
                generated_next_chunk = False
                tiles_objects = tiles_objects[1:] + [new_tile_obs]
                new_tile_obs = {1: [], 2: [], 3: [], 4: []}
    else:
        print('Failed connecting to remote API server')
    print('Program ended')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--ip', type=str, default='127.0.0.1')
    ip = argparser.parse_args().ip

    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart(ip, 19997, True, True,
                             5000, 5)  # Connect to CoppeliaSim
    try:
        main(clientID)
    except KeyboardInterrupt:
        print("Interrupted!")
        # Stop simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        sim.simxFinish(clientID)
