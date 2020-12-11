# SLAM Controller

import math
import random
import copy
from controller import Robot, Motor, DistanceSensor
import SLAM_controller_supervisor
import numpy as np
import collections

state = "move" # Drive along the course
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

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
GROUND_SENSOR_THRESHOLD = 600 # Light intensity units
LIDAR_SENSOR_MAX_RANGE = 3. # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

#RANSAC values
MAX_TRIALS = 100 # Max times to run algorithm
MAX_SAMPLE = 8 # Randomly select X points
MIN_LINE_POINTS = 5 # If less than 5 points left, stop algorithm
RANSAC_TOLERANCE = 0.05 # If point is within 5 cm of line, it is part of the line
RANSAC_CONSENSUS = 5 # At least 5 points required to determine if a line

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
EPUCK_AXLE_DIAMETER = 0.053
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
ground_sensor_readings = [0, 0, 0]
ground_sensors = [robot.getDistanceSensor('gs0'), robot.getDistanceSensor('gs1'), robot.getDistanceSensor('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

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
n = 8 # number of static landmarks
mu = []
cov = []

#Stored global [x, y, j] for observed landmarks to check if the landmark has been seen before (is within 0.05 cm radius of previous x, y)
landmark_globals = []

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
        return None
    
    #Lidar centered at robot 0,0 so no translation needed
    #Convert lidar -> robot adding math.pi/2 to fix direction
    bQ_x = math.sin( lidar_bin + math.pi/2) * lidar_distance
    bQ_y = math.cos( lidar_bin + math.pi/2) * lidar_distance
    #print(bQ_x, bQ_y)
    #convert robot -> world
    x = math.cos( pose_theta ) * bQ_x - math.sin( pose_theta ) * bQ_y + pose_x
    y = math.sin( pose_theta ) * bQ_x + math.cos( pose_theta ) * bQ_y + pose_y
    #print(x,y)
    return [x, y]


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
       
    return phi_l_pct, phi_r_pct, (phi_l, phi_r)


def perform_least_squares_line_estimate(lidar_world_coords, selected_points):
    """
    @param lidar_world_coords: List of world coordinates read from lidar data (tuples of the form [x, y])
    @param selected_points: Indicies of the points selected for this least squares line estimation
    @return m, b: Slope and intercept for line estimated from data - y = mx + b
    """
    sum_y = sum_yy = sum_x = sum_xx = sum_yx = 0 #Sums of y coordinates, y^2 for each coordinate, x coordinates, x^2 for each coordinate, and y*x for each point

    for point in selected_points:
        world_coord = lidar_world_coords[point]

        sum_y += world_coord[1]
        sum_yy += world_coord[1]**2
        sum_x += world_coord[0]
        sum_xx += world_coord[0]**2
        sum_yx += world_coord[0] * world_coord[1]

    num_points = len(selected_points)
    b = (sum_y*sum_xx - sum_x*sum_yx) / (num_points*sum_xx - sum_x**2)
    m = (num_points*sum_yx - sum_x*sum_y) / (num_points*sum_xx - sum_x**2)

    return m, b

def distance_to_line(x, y, m, b):
    """
    @param x: x coordinate of point to find distance to line from
    @param y: y coordinate of point to find distance to line from
    @param m: slope of line
    @param b: intercept of line
    @return dist: the distance from the given point coordinates to the given line
    """

    #Line perpendicular to input line crossing through input point - m*m_o = -1 and y=m_o*x + b_o => b_o = y - m_o*x
    m_o = -1.0/m
    b_o = y - m_o*x

    #Intersection point between y = m*x + b and x = m_o*x + b_o
    p_x = (b - b_o) / (m_o - m)
    p_y = ((m_o * (b - b_o)) / (m_o - m)) + b_o

    return math.dist([x,y], [p_x,p_y])
        


def get_line_landmark(line):
    global mu
    #slope perpendicular to input line
    m_o = -1.0 / line[0]

    #landmark position
    lm_x = line[1] / (m_o - line[0])
    lm_y = (m_o*line[1]) / (m_o - line[0])
    lm_j = 0

    found = False
    for [x, y, j] in landmark_globals:
        if math.dist([x, y], [lm_x, lm_y]) <= 0.05:
            lm_j = j
            found = True
            break
        else:
            lm_j += 1

    #If we didn't match the landmark to a previously found one and we're over the cap for new landmarks, return none to not calculate with this landmark
    if not found and len(landmark_globals) >= n:
        return None
    #Otherwise, add it to our landmarks
    elif not found and len(landmark_globals) < n:
        landmark_globals.append([lm_x, lm_y, lm_j])

    #convert to robot-relative positioning with radius from the robot and theta relative to robot
    r = math.dist([lm_x, lm_y], [mu[0][0], mu[1][0]])
    theta = math.atan2(lm_x, lm_y)
    theta = theta - mu[2][0]

    return [r, theta, lm_j]

def extract_line_landmarks(lidar_world_coords):
    """
    @param lidar_world_coords: List of world coordinates read from lidar data (tuples of the form [x, y])
    @return found_landmarks: list of landmarks found through the RANSAC done on the lidar data
    """

    found_lines = [] #list of tuples of the form [m, b] of detected lines

    linepoints = [] #list of laser data points not yet associated to a found line

    found_landmarks = [] #list to keep track of found landmarks from lines, stored as [r, theta, j] relative to robot

    for i in range(len(lidar_world_coords)):
        linepoints.append(i)
    
    num_trials = 0
    while (num_trials < MAX_TRIALS and len(linepoints) >= MIN_LINE_POINTS):
        rand_selected_points = []

        #randomly choose up to MAX_SAMPLE points for the least squares
        for i in range(min(MAX_SAMPLE, len(linepoints))):
            temp = -1
            new_point = False
            while not new_point:
                temp = random.randint(0, len(linepoints)-1) #generate a random integer between 0 and our total number of remaining line points to choose from
                if linepoints[temp] not in rand_selected_points:
                    new_point = True
            rand_selected_points.append(linepoints[temp])
        
        #Now compute a line based on the randomly selected points
        m, b = perform_least_squares_line_estimate(lidar_world_coords, rand_selected_points)

        consensus_points = [] # points matching along the found line
        new_linepoints = [] # points not matching along the found line, if we say the line is a landmark, these are our new set of unmatched points

        for point in linepoints:
            curr_point = lidar_world_coords[point]
            #distance to line from the point
            dist = distance_to_line(curr_point[0], curr_point[1], m, b)

            if dist < RANSAC_TOLERANCE:
                consensus_points.append(point)
            else:
                new_linepoints.append(point)
        
        if len(consensus_points) >= RANSAC_CONSENSUS:
            #Calculate an updated line based on every point within the consensus
            m, b = perform_least_squares_line_estimate(lidar_world_coords, consensus_points)

            #add to found lines
            found_lines.append([m,b])

            #rewrite the linepoints as the linepoints that didn't match with this line to only search unmatched points for lines
            linepoints = new_linepoints.copy()

            #restart number of trials
            num_trials = 0
        else:
            num_trials += 1
    
    #Now we'll calculate the point closest to the origin for each line found and add these as found landmarks
    for line in found_lines:
        new_landmark = get_line_landmark(line)
        if new_landmark is not None:
            found_landmarks.append(new_landmark)
    
    return found_landmarks

def EKF_init(x_init):
    global Rt, Qt, mu, cov

    Rt = 5*np.array([[0.01,0,0],
               [0,0.01,0],
               [0,0,0.01]])
    Qt = np.array([[0.01,0],
               [0,0.01]])

    mu = np.append(np.array([x_init]).T,np.zeros((2*n,1)),axis=0)

    cov = 1e6*np.eye(2*n+3)

    #Init robot to known truth pos
    cov[:3,:3] = np.eye(3,3)*np.array(x_init).T

def EKF_predict(u, Rt):
    #global mu
    #n = len(mu)

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


def EKF_update(obs, Qt):
    
    for [r, theta, j] in obs:
        j = int(j)
        #Landmark has not been observed before
        if cov[2*j+3][2*j+3] >= 1e6 and cov[2*j+4][2*j+4] >= 1e6:
            # define landmark estimate as current measurement
            mu[2*j+3][0] = mu[0][0] + r*np.cos(theta+mu[2][0])
            mu[2*j+4][0] = mu[1][0] + r*np.sin(theta+mu[2][0])

        #Landmark has been seen before, use past values
        # compute expected observation
        delta = np.array([mu[2*j+3][0] - mu[0][0], mu[2*j+4][0] - mu[1][0]])
        q = delta.T.dot(delta)
        sq = np.sqrt(q)
        z_theta = np.arctan2(delta[1],delta[0])
        z_hat = np.array([[sq], [z_theta-mu[2][0]]])
        
        # calculate Jacobian
        F = np.zeros((5,n))
        F[:3,:3] = np.eye(3)
        F[3,2*j+3] = 1
        F[4,2*j+4] = 1
        H_low = np.array([[-sq*delta[0], -sq*delta[1], 0, sq*delta[0], sq*delta[1]],
                        [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
        H = 1/q*H_low.dot(F)

        # calculate Kalman gain        
        K = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov).dot(H.T)+Qt))
        
        # calculate difference between expected and real observation
        z_dif = np.array([[r],[theta]])-z_hat
        z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi
        
        # update state vector and covariance matrix        
        mu_new = mu + K.dot(z_dif)
        cov_new = (np.eye(n)-K.dot(H)).dot(cov)
    
    print('Updated location\t x: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}'.format(mu[0][0],mu[1][0],mu[2][0]))
    return mu_new, cov_new

def move(target_pose):
    '''
    motion_noise = np.matmul(np.random.randn(1,3), Rt)[0]
    [dtrans, drot1, drot2] = u[:3] + motion_noise
    
    x = [pose_x, pose_y, pose_theta]
    x_new = x[0] + dtrans*np.cos(x[2]+drot1)
    y_new = x[1] + dtrans*np.sin(x[2]+drot1)
    theta_new = (x[2] + drot1 + drot2 + np.pi) % (2*np.pi) - np.pi
    
    x_true = [x_new, y_new, theta_new]
    '''
    lspeed, rspeed, (phi_l, phi_r) = get_wheel_speeds(target_pose)
    dtrans = np.linalg.norm(target_pose[:2] - np.array(mu[0][0],mu[1][0]))
    u = [dtrans, phi_l, phi_r]
    
    print("lspeed: ", lspeed, "rspeed: ", rspeed)
    leftMotor.setVelocity(lspeed)
    rightMotor.setVelocity(rspeed)

    return u

def generate_obs():
    lidar_data = lidar.getRangeImage()

    #convert lidar to world locations
    lidar_readings = []
    for i in range(21):
        lidar_found_loc = convert_lidar_reading_to_world_coord(i, lidar_data[i])
        if lidar_found_loc is not None:
            lidar_readings.append(lidar_found_loc)

    #Run RANSAC on lidar_data
    obs = extract_line_landmarks(lidar_readings)

    return obs #[r, theta, j]

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
    global robot, ground_sensors, ground_sensor_readings, pose_x, pose_y, pose_theta, state
    global leftMotor, rightMotor, SIM_TIMESTEP, WHEEL_FORWARD, WHEEL_STOPPED, WHEEL_BACKWARDS
    global cov, Rt, Qt, mu
    
    last_odometry_update_time = None

    #path for robot to try and follow
    robot_path = [[ 0.30928,   0.176902,   1.046032],
                  [ 0.433827,  0.642393,   1.3084],
                  [ 0.434453,  1.702051,   1.5702],
                  [ 0.275088,  1.978296,   2.353669],
                  [-0.144059,  1.9779351,  3.139076],
                  [-0.484122,  1.556442,   4.449154],
                  [-0.486042,  0.158664,   4.711667],
                  [-0.345818,  0.018178,  -0.783037],
                  [-0.03,     -0.05,       0]         ]
    pos_idx = 0
    
    # Keep track of which direction each wheel is turning
    left_wheel_direction = WHEEL_STOPPED
    right_wheel_direction = WHEEL_STOPPED
    
    # Sensor burn-in period
    for i in range(10): robot.step(SIM_TIMESTEP)

    #Known truth start pose
    start_pose = SLAM_controller_supervisor.supervisor_get_robot_pose()
    pose_x, pose_y, pose_theta = start_pose

    lidar_obs = []
    u = [0, 0, 0] #State vector
    #Tolerances for reaching waypoint state
    x_tol = 0.05
    y_tol = 0.05
    theta_tol = 0.05

    #Init with known start location
    EKF_init(start_pose)

    print("MU: ", mu)

    print("Cov: ", cov)

    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:
        # Read ground sensor values
        for i, gs in enumerate(ground_sensors):
            ground_sensor_readings[i] = gs.getValue()
        loop_closure_detection_time = 0
        
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()
        print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))

        #Get next position in path to traverse
        if pos_idx == len(robot_path):
            pos_idx = 0
        target_pose = robot_path[pos_idx]

        #Move
        if state == "move":
            #Move and update state vector for EKF
            u = move(target_pose)
            print("U: ", u)

            #At waypoint location
            if (abs(pose_x - target_pose[0]) < x_tol and 
                abs(pose_y - target_pose[1]) < y_tol and
                abs(pose_theta - target_pose[2]) < theta_tol): 
                
                print("\n****At Waypoint****\n")
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                pos_idx += 1
                state = 'sense'
        #Sense
        elif state == "sense":
            lidar_obs = generate_obs()
            state = 'predict'

        #Predict
        elif state == "predict":
            mu_new, cov = EKF_predict(u, Rt)
            mu = np.append(mu,mu_new,axis=1)
            state = 'update'
        #Update
        elif state == "update":
            mu_new, cov = EKF_update(lidar_obs, Qt)
            mu = np.append(mu,mu_new,axis=1)
            state = 'move'
        else:
            #Stop
            print("Stopping!")
            left_wheel_direction, right_wheel_direction = 0, 0
            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)    
            break

        # Loop Closure
        if False not in [ground_sensor_readings[i] < GROUND_SENSOR_THRESHOLD for i in range(3)]:
            loop_closure_detection_time += SIM_TIMESTEP / 1000.
            if loop_closure_detection_time > 0.1:
                pose_x, pose_y, pose_theta = SLAM_controller_supervisor.supervisor_get_robot_pose()
                #Update current location to ground truth
                u = move([pose_x, pose_y, pose_theta])
                mu[0][0], mu[1][0], mu[2][0] = pose_x, pose_y, pose_theta
                state = 'sense'
                loop_closure_detection_time = 0
        else:
            loop_closure_detection_time = 0

if __name__ == "__main__":
    main()
    
    
    
    
    
    