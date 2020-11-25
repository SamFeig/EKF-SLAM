# SLAM Controller

from controller import Robot, Motor

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
GROUND_SENSOR_THRESHOLD = 600 # Light intensity units
LIDAR_SENSOR_MAX_RANGE = 3. # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

# Pose values to be updated by solving odometry equations
pose_x = 0
pose_y = 0
pose_theta = 0

#velocity reduction percent
MAX_VEL_REDUCTION = 0.8 # Run robot at 20% of max speed

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = .125 * MAX_VEL_REDUCTION # To be filled in with ePuck wheel speed in m/s

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)


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
    
    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:   
        
        # If first time entering the loop, just take the current time as "last odometry update" time
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
    
    
        # Update Odometry
        time_elapsed = robot.getTime() - last_odometry_update_time
        last_odometry_update_time += time_elapsed
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        
        left_wheel_direction, right_wheel_direction = WHEEL_FORWARD, WHEEL_FORWARD
        leftMotor.setVelocity(leftMotor.getMaxVelocity())
        rightMotor.setVelocity(rightMotor.getMaxVelocity())
    
        print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))



if __name__ == "__main__":
    main()
    
    
    
    
    
    