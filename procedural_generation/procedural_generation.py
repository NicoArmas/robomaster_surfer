try:
    import sim
except:
    pass
import time

from mazeGenerator import generate_lab, state_from_row

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True,
                         5000, 5)  # Connect to CoppeliaSim


def place_obstacle(clientID, handle, position):
    _, a = sim.simxCopyPasteObjects(
        clientID, [handle], sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(
        clientID, a[0], -1, position, sim.simx_opmode_oneshot)
    return a[0]


def remove_obstacle(clientID, handle):
    sim.simxRemoveObject(
        clientID, handle, sim.simx_opmode_oneshot)


def move_obstacle(clientID, handle, position):
    sim.simxSetObjectPosition(
        clientID, handle, -1, position, sim.simx_opmode_oneshot)
    return handle


def place_obstacles(obs, handlers, switcher_count, offset_world, offset_blocks, obstacle_size):
    tile_obstacles = {1: [], 2: [], 3: [], 4: []}
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            if obs[i][j] != 0:
                handle = handlers[obs[i][j]-1]
                position = (switcher_count+offset_world-offset_blocks+i *
                            obstacle_size, j*obstacle_size+offset_blocks, 0.2)
                if obs[i][j] != 1:
                    position = (switcher_count+offset_world-offset_blocks+i *
                            obstacle_size, j*obstacle_size+offset_blocks, 0.002)
                tile_obstacles[obs[i][j]].append(
                    place_obstacle(clientID, handle, position))

    return tile_obstacles


if clientID != -1:
    # Get Handlers of important objects
    _, hanlder_RM = sim.simxGetObjectHandle(
        clientID, "/RoboMaster", sim.simx_opmode_blocking)

    _, handler_camera = sim.simxGetObjectHandle(
        clientID, "/DefaultCamera", sim.simx_opmode_blocking)

    _, hanlder_obstacle = sim.simxGetObjectHandle(
        clientID, "/ConcretBlock", sim.simx_opmode_blocking)

    _, handler_coin1 = sim.simxGetObjectHandle(
        clientID, "/Coin1", sim.simx_opmode_blocking)

    _, handler_coin2 = sim.simxGetObjectHandle(
        clientID, "/Coin2", sim.simx_opmode_blocking)

    _, handler_coin5 = sim.simxGetObjectHandle(
        clientID, "/Coin5", sim.simx_opmode_blocking)

    handler_blocks = [hanlder_obstacle,
                      handler_coin1, handler_coin2, handler_coin5]

    _, hanlder_wall_right = sim.simxGetObjectHandle(
        clientID, "/240cmHighWall400cm[1]", sim.simx_opmode_blocking)

    _, hanlder_wall_left = sim.simxGetObjectHandle(
        clientID, "/240cmHighWall400cm[0]", sim.simx_opmode_blocking)

    _, pos_camera = sim.simxGetObjectPosition(
        clientID, handler_camera, -1, sim.simx_opmode_blocking)

    _, pos_RM = sim.simxGetObjectPosition(
        clientID, hanlder_RM, -1, sim.simx_opmode_blocking)

    pose_RM_camera_relation = (
        pos_RM[0]-pos_camera[0], pos_camera[1], pos_camera[2])

    tiles_handlers = []
    switcher_count = 0  # (number of created tiles * length tile)

    corridor_size = 1.5
    obstacle_size = corridor_size/3
    offset_blocks = obstacle_size/2

    for i in range(5):
        # Save tiles handlers
        _, id = sim.simxGetObjectHandle(
            clientID, "/ResizableFloorMedium/element["+str(i)+"]", sim.simx_opmode_blocking)
        tiles_handlers.append(id)
    cur_tile_x = pos_RM[0]

    _, pos_tile = sim.simxGetObjectPosition(
        clientID, tiles_handlers[0], -1, sim.simx_opmode_blocking)
    pos_tile = (pos_tile[0]+20, pos_tile[1], pos_tile[2])

    # Get walls position
    _, pos_wall = sim.simxGetObjectPosition(
        clientID, hanlder_wall_left, -1, sim.simx_opmode_blocking)

    offset_world = round(pos_RM[0])

    def update_tiles(tile_handler, pos_tile, wall_pose):
        # Set walls new position
        wlp = (wall_pose[0]+5, wall_pose[1], wall_pose[2])
        tp = (pos_tile[0]+5, pos_tile[1], pos_tile[2])
        sim.simxSetObjectPosition(
            clientID, hanlder_wall_left, -1, wlp, sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(
            clientID, hanlder_wall_right, -1,  (wall_pose[0]+5, wall_pose[1]+1.5, wall_pose[2]), sim.simx_opmode_blocking)
        # Set tile new position
        sim.simxSetObjectPosition(
            clientID, tile_handler, -1, tp, sim.simx_opmode_blocking)

        return tp, wlp

    def update_camera_pose(pose):
        tmp = (pose[0]-pose_RM_camera_relation[0],
               pose_RM_camera_relation[1],
               pose_RM_camera_relation[2])
        return tmp

    def from_map_to_pos_list(obs, switcher_count):
        obstacle_positions = {1: [], 2: [], 3: [], 4: []}
        for i in range(len(obs)):
            for j in range(len(obs[0])):
                if obs[i][j] != 0:
                    if obs[i][j]==1:
                        obstacle_positions[obs[i][j]].append((switcher_count+offset_world-offset_blocks+i *
                                                          obstacle_size, j*obstacle_size+offset_blocks, 0.2))
                    else:
                        obstacle_positions[obs[i][j]].append((switcher_count+offset_world-offset_blocks+i *
                                                          obstacle_size, j*obstacle_size+offset_blocks, 0.02))
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

        if not start and abs(cur_tile_x - pos_RM[0]) > 5:
            start = True
            moved_tile = tiles_handlers[0]
            pos_tile, pos_wall = update_tiles(
                moved_tile, pos_tile, pos_wall)
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
                    clientID, handler_blocks[el-1], obstacle_pos))

            elif start and len(tiles_objects[0][el]) > 0:
                handle_to_move = tiles_objects[0][el][0]
                tiles_objects[0][el] = tiles_objects[0][el][1:]
                remove_obstacle(clientID, handle_to_move)

        if len(pos_obstacles[1]) == 0 and len(tiles_objects[0][1]) == 0:
            start = False
            generated_next_chunk = False
            tiles_objects = tiles_objects[1:]+[new_tile_obs]
            new_tile_obs = {1: [], 2: [], 3: [], 4: []}

        _, pos_RM = sim.simxGetObjectPosition(
            clientID, hanlder_RM, -1, sim.simx_opmode_blocking)
        #pos_RM = (pos_RM[0]+0.3, pos_RM[1], pos_RM[2])
        #sim.simxSetObjectPosition(
        #    clientID, hanlder_RM, -1, pos_RM, sim.simx_opmode_blocking)

        pose = update_camera_pose(pos_RM)
        sim.simxSetObjectPosition(
            clientID, handler_camera, -1, pose, sim.simx_opmode_blocking)
        #time.sleep(0.08)

else:
    print('Failed connecting to remote API server')
print('Program ended')
