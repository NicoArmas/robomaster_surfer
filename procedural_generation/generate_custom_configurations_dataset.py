try:
    import sim
except:
    pass

configurations = [(0, 1, 1, 1), (0, 1, 1, 2), (0, 0, 1, 2), (1, 0, 0, 0),
                  (1, 1, 0, 0), (1, 1, 0, 1), (1, 0, 1, 0), (1, 0, 1, 2)]


print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True,
                           5000, 5)  # Connect to CoppeliaSim



z = 0.2


def place_obstacle(clientID, handle, position):
    _, a = sim.simxCopyPasteObjects(
        clientID, [handle], sim.simx_opmode_blocking)
    a = a[0]
    sim.simxSetObjectIntParameter(
        clientID, a, 3004, 0, sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(
        clientID, a, -1, position, sim.simx_opmode_blocking)
    sim.simxSetObjectIntParameter(
        clientID, a, 3004, 1, sim.simx_opmode_blocking)
    return a


def remove_obstacle(clientID, handle):
    sim.simxRemoveObject(
        clientID, handle, sim.simx_opmode_oneshot)


def move_obstacle(clientID, handle, position):
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 0, sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(
        clientID, handle, -1, position, sim.simx_opmode_blocking)
    sim.simxSetObjectIntParameter(
        clientID, handle, 3004, 1, sim.simx_opmode_blocking)
    # handle2 = place_obstacle(clientID, handle, position)
    # remove_obstacle(clientID, handle)
    return handle

def create_custom_configuration(configuration, handler_RM):
    c = -0.5
    for i in range(3):
        if configuration[i] == 1:
            obstacle_pos = (pos_RM[0]+1.5, pos_RM[1]+c+0.5*i, z)
            place_obstacle(clientID, hanlder_obstacle, obstacle_pos)

    pos = (pos_RM[0], pos_RM[1]+c+0.5*configuration[3], pos_RM[2])
    print(pos)
    move_obstacle(clientID, handler_RM, pos)



if clientID != -1:

    # Get Handlers of important objects
    _, hanlder_RM = sim.simxGetObjectHandle(
        clientID, "/RoboMaster", sim.simx_opmode_blocking)

    _, hanlder_obstacle = sim.simxGetObjectHandle(
        clientID, "/Obstacle", sim.simx_opmode_blocking)

    _, pos_RM = sim.simxGetObjectPosition(
        clientID, hanlder_RM, -1, sim.simx_opmode_blocking)

    create_custom_configuration(configurations[2], hanlder_RM)


else:
    print('Failed connecting to remote API server')
print('Program ended')
