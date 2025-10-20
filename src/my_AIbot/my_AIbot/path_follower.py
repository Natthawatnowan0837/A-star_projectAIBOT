#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from tf_transformations import euler_from_quaternion


class GoToPath(Node):
    def __init__(self):
        super().__init__('go_to_path')

        # Publisher / Subscriber
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/astar_path', self.path_callback, 10)

        # Robot state variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.path = []          # List of waypoints (x, y)
        self.current_target = 0 # Current waypoint index
        self.reached = False

        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("GoToPath node started. Waiting for path on topic /astar_path.")

    # ====== Odometry callback ======
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.yaw = yaw

    # ====== Path callback ======
    def path_callback(self, msg: Path):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_target = 0
        self.reached = False
        self.get_logger().info(f"Received path with {len(self.path)} waypoints.")

    # ====== Control loop ======
    def control_loop(self):
        if not self.path or self.reached:
            return

        # Current waypoint target
        goal_x, goal_y = self.path[self.current_target]

        # Compute distance and heading error
        inc_x = goal_x - self.x
        inc_y = goal_y - self.y
        distance = math.sqrt(inc_x ** 2 + inc_y ** 2)
        angle_to_goal = math.atan2(inc_y, inc_x)
        angle_diff = math.atan2(math.sin(angle_to_goal - self.yaw),
                                math.cos(angle_to_goal - self.yaw))

        cmd = Twist()

        # Check if reached current waypoint
        if distance < 0.1:
            self.current_target += 1
            if self.current_target >= len(self.path):
                self.get_logger().info("Path completed successfully.")
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self.reached = True
                return
            else:
                self.get_logger().info(f"Moving to next waypoint ({self.current_target}/{len(self.path)}).")
                return

        # Rotation control
        if abs(angle_diff) > 0.2:
            cmd.angular.z = 0.5 * angle_diff
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.15
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GoToPath()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
