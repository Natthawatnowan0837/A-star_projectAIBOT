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
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/astar_path', self.path_callback, 10)

        # Robot state variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.path = []
        self.current_target = 0
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

        goal_x, goal_y = self.path[self.current_target]
        inc_x = goal_x - self.x
        inc_y = goal_y - self.y
        distance = math.hypot(inc_x, inc_y)
        angle_to_goal = math.atan2(inc_y, inc_x)
        angle_diff = math.atan2(math.sin(angle_to_goal - self.yaw),
                                math.cos(angle_to_goal - self.yaw))

        cmd = Twist()

        if distance < 0.1:
            self.current_target += 1
            if self.current_target >= len(self.path):
                self.get_logger().info("Path completed successfully.")
                self.stop_robot()
                self.reached = True
                return
            else:
                self.get_logger().info(f"Moving to next waypoint ({self.current_target}/{len(self.path)}).")
                return

        # === Speed limits ===
        MAX_LIN_SPEED = 0.35
        MAX_ANG_SPEED = 1.2

        if abs(angle_diff) > 0.4:
            cmd.angular.z = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, 1.5 * angle_diff))
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = MAX_LIN_SPEED * (1.0 - abs(angle_diff))
            cmd.angular.z = max(-MAX_ANG_SPEED, min(MAX_ANG_SPEED, 1.0 * angle_diff))

        self.cmd_pub.publish(cmd)

    # ====== Stop robot safely ======
    def stop_robot(self):
        # 1. หยุดความเร็ว
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info("Robot stopped safely.")

        # 2. รีเซ็ตค่า odometry ให้เป็นศูนย์
        odom = Odometry()
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.orientation.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)
        self.get_logger().info("Odometry reset to zero.")


def main(args=None):
    rclpy.init(args=args)
    node = GoToPath()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node.get_logger().info("KeyboardInterrupt detected (Ctrl+C). Stopping robot and resetting odometry...")
            node.stop_robot()
        except Exception as e:
            # ไม่ log ผ่าน ROS (เพราะ context อาจถูกปิดแล้ว)
            print(f"[WARN] Failed to stop robot cleanly: {e}")
    finally:
        # ปิด node อย่างปลอดภัย แต่ไม่เรียก shutdown ซ้ำ
        try:
            node.destroy_node()
        except Exception:
            pass




if __name__ == '__main__':
    main()
