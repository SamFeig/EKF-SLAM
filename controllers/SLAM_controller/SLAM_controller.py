# SLAM Controller

import math
import copy
from controller import Robot, Motor, DistanceSensor
import SLAM_controller_supervisor
import numpy as np
import collections

state = "line_follower" # Drive along the course
USE_ODOMETRY = False # False for ground truth pose information, True for real odometry
# create the Robot instance.
SLAM_controller_supervisor.init_supervisor()
robot = SLAM_controller_supervisor.supervisor

# Map Variables
MAP_BOUNDS = [1.,1.] 
CELL_RESOLUTIONS = np.array([0.1, 0.1]) # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

def populate_map(m):
    obs_list = SLAM_controller_supervisor.supervisor_get_obstacle_positions()
    obs_size = 0.06 # 6cm boxes
    for obs in obs_list:
        obs_coords_lower = obs - obs_size/2.
        obs_coords_upper = obs + obs_size/2.
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1
        obs_coords_lower = [obs[0] - obs_size/2, obs[1] + obs_size/2.]
        obs_coords_upper = [obs[0] + obs_size/2., obs[1] - obs_size/2.]
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
GROUND_SENSOR_THRESHOLD = 600 # Light intensity units
LIDAR_SENSOR_MAX_RANGE = 3. # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3

MAX_VEL_REDUCTION = 0.25
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION # m/s
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# create the Robot instance.
#robot = Robot()

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# get and enable lidar 
lidar = robot.getLidar("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

#Initialize lidar motors
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

lidar_data = np.zeros(LIDAR_ANGLE_BINS)
print(lidar_data)

step = LIDAR_ANGLE_RANGE/LIDAR_ANGLE_BINS

lidar_offsets = np.linspace(-step*(LIDAR_ANGLE_BINS//2), step*(LIDAR_ANGLE_BINS//2), LIDAR_ANGLE_BINS)
#lidar_offsets = -1 * lidar_offsets
print(lidar_offsets)

#EKF Vars
n = 50 # number of static landmarks
mu = []
mu_new = []
cov = []
c_prob = []

#From Lab 4
def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    """
    @param lidar_bin: The beam index that provided this measurement
    @param lidar_distance: The distance measurement from the sensor for that beam
    @return world_point: List containing the corresponding (x,y) point in the world frame of reference
    """
    
    # YOUR CODE HERE
    #print("Dist:", lidar_distance, "Ang:", lidar_bin)
    
    #No detection
    if(lidar_distance > LIDAR_SENSOR_MAX_RANGE):# or lidar_distance > math.sqrt(2)):
        return
    
    #Lidar centered at robot 0,0 so no translation needed
    #Convert lidar -> robot adding math.pi/2 to fix direction
    bQ_x = math.sin( lidar_bin + math.pi/2) * lidar_distance
    bQ_y = math.cos( lidar_bin + math.pi/2) * lidar_distance
    #print(bQ_x, bQ_y)
    #convert robot -> world
    x = math.cos( pose_theta ) * bQ_x - math.sin( pose_theta ) * bQ_y + pose_x
    y = math.sin( pose_theta ) * bQ_x + math.cos( pose_theta ) * bQ_y + pose_y
    #print(x,y)
    return x, y

#From Lab 5
def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))

#From Lab 5
def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None
    
    return np.array([(col+0.5)*CELL_RESOLUTIONS[1], (row+0.5)*CELL_RESOLUTIONS[0]])


#From Lab 5
def get_wheel_speeds(target_pose):
    '''
    @param target_pose: Array of (x,y,theta) for the destination robot pose
    @return motor speed as percentage of maximum for left and right wheel motors
    '''
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = SLAM_controller_supervisor.supervisor_get_robot_pose()


    bearing_error = math.atan2( (target_pose[1] - pose_y), (target_pose[0] - pose_x) ) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x,pose_y]))
    heading_error = target_pose[2] -  pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0
    
    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer
    
    if distance_error < 0.05 and abs(heading_error) < 0.05:    
        left_speed_pct = 0
        right_speed_pct = 0
        
    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()
       
    return phi_l_pct, phi_r_pct

def EKF_init(x_init):
    global Rt, Qt, mu, mu_new, cov, c_prob

    Rt = 5*np.array([[0.1,0,0],
               [0,0.01,0],
               [0,0,0.01]])
    Qt = np.array([[0.01,0],
               [0,0.01]])

    mu = np.append(np.array([x_init]).T,np.zeros((2*n,1)),axis=0)
    mu_new = mu

    cov = float("inf")*np.eye(2*n+3)
    cov[:3,:3] = np.zeros((3,3))

    c_prob = 0.5*np.ones((n,1))

def EKF_predict(u, Rt):
    n = len(mu)

    # Define motion model f(mu,u)
    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans*np.cos(mu[2][0]+drot1)],
                       [dtrans*np.sin(mu[2][0]+drot1)],
                       [drot1 + drot2]])
    F = np.append(np.eye(3),np.zeros((3,n-3)),axis=1)
    
    # Predict new state
    mu_bar = mu + (F.T).dot(motion)
    
    # Define motion model Jacobian
    J = np.array([[0,0,-dtrans*np.sin(mu[2][0]+drot1)],
                  [0,0,dtrans*np.cos(mu[2][0]+drot1)],
                  [0,0,0]])
    G = np.eye(n) + (F.T).dot(J).dot(F)
    
    # Predict new covariance
    cov_bar = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)
    
    print('Predicted location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu_bar[0][0],mu_bar[1][0],mu_bar[2][0]))
    return mu_bar, cov_bar


def EKF_update(obs, c_prob, Qt):
    N = len(mu)
    
    for [r, theta, j] in obs:
        j = int(j)
        # if landmark has not been observed before
        if cov[2*j+3][2*j+3] >= 1e6 and cov[2*j+4][2*j+4] >= 1e6:
            # define landmark estimate as current measurement
            mu[2*j+3][0] = mu[0][0] + r*np.cos(theta+mu[2][0])
            mu[2*j+4][0] = mu[1][0] + r*np.sin(theta+mu[2][0])

        
        # if landmark is static
        if c_prob[j] >= 0.5:
            # compute expected observation
            delta = np.array([mu[2*j+3][0] - mu[0][0], mu[2*j+4][0] - mu[1][0]])
            q = delta.T.dot(delta)
            sq = np.sqrt(q)
            z_theta = np.arctan2(delta[1],delta[0])
            z_hat = np.array([[sq], [z_theta-mu[2][0]]])
            
            # calculate Jacobian
            F = np.zeros((5,N))
            F[:3,:3] = np.eye(3)
            F[3,2*j+3] = 1
            F[4,2*j+4] = 1
            H_z = np.array([[-sq*delta[0], -sq*delta[1], 0, sq*delta[0], sq*delta[1]],
                            [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
            H = 1/q*H_z.dot(F)
    
            # calculate Kalman gain        
            K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T)+Qt))
            
            # calculate difference between expected and real observation
            z_dif = np.array([[r],[theta]])-z_hat
            z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi
            
            # update state vector and covariance matrix        
            mu = mu + K.dot(z_dif)
            cov = (np.eye(N)-K.dot(H)).dot(cov)
    
    print('Updated location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu[0][0],mu[1][0],mu[2][0]))
    return mu, cov, c_prob

def move(u):
    lspeed, rspeed = get_wheel_speeds(u)
    print("lspeed: ", lspeed, "rspeed: ", rspeed)
    leftMotor.setVelocity(lspeed)
    rightMotor.setVelocity(rspeed)

def sense(lt):
    lidar_data = lidar.getRangeImage()
    print(lidar_data)
    #update_map(lidar_data)

def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
    '''
    Given the amount of time passed and the direction each wheel was rotating,
    update the robot's pose information accordingly
    '''
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    # Update pose_theta
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    #Update pose_x
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    #Update pose_y
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    
    
def main():
    global robot, pose_x, pose_y, pose_theta
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    
    last_odometry_update_time = None
    
    # Keep track of which direction each wheel is turning
    left_wheel_direction = WHEEL_STOPPED
    right_wheel_direction = WHEEL_STOPPED
    
    # Sensor burn-in period
    for i in range(10): robot.step(SIM_TIMESTEP)

    start_pose = SLAM_controller_supervisor.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = start_pose

    #Init
    EKF_init(start_pose)

    print("MU: ", mu)

    print("Cov: ", cov)

    print("c_prob: ", c_prob)

    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:   
        
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()
        print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))

        #Move
        move(u)

        #Sense
        sense()


        #Predict
        mu_new, cov = EKF_predict(u, Rt)
        mu = np.append(mu,mu_new,axis=1)
        #Update
        mu_new, cov, c_prob_new = EKF_update(lidar_obs, c_prob[:,-1].reshape(n,1), Qt)
        mu = np.append(mu,mu_new,axis=1)
        c_prob = np.append(c_prob, c_prob_new, axis=1)

        #Plot stuffs

if __name__ == "__main__":
    main()
    
    
    
    
    
    