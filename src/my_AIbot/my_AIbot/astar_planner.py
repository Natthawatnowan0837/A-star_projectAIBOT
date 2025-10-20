#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
from tf_transformations import euler_from_quaternion


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        # === Subscribers ===
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # === Publishers ===
        self.map_pub = self.create_publisher(OccupancyGrid, '/ocgm', 10)
        self.path_pub = self.create_publisher(Path, '/astar_path', 10)

        # === Map parameters ===
        self.map_size = 10.0      # meters
        self.resolution = 0.05    # m/cell
        self.width = int(self.map_size / self.resolution)
        self.height = int(self.map_size / self.resolution)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.memory = np.zeros_like(self.grid)

        # === Start and Goal positions ===
        self.start_world = (0.0, 0.0)
        self.goal_world = (10.0, 0.0)  # Target far away

        # === Robot state ===
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_ready = False

        # === Flags ===
        self.path_published = False
        self.last_plan_time = self.get_clock().now()
        self.path = []

        # === Replan tracking ===
        self.last_replan_pose = (0.0, 0.0)
        self.replan_move_thresh = 0.5  # replan every 0.5 m moved

        self.get_logger().info("AStarPlanner node started successfully.")

    # ========== ODOMETRY ==========
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw
        self.odom_ready = True

    # ========== LASER SCAN ==========
    def laser_callback(self, msg: LaserScan):
        if not self.odom_ready:
            return

        # Decay map memory
        self.memory = np.clip(self.memory * 0.97, 0, 100)

        obstacle_detected = False
        angle = msg.angle_min

        for r in msg.ranges:
            if np.isnan(r) or np.isinf(r) or r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            if r < 0.25:
                obstacle_detected = True

            # LiDAR → world
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            wx = self.robot_x + (lx * math.cos(self.robot_yaw) - ly * math.sin(self.robot_yaw))
            wy = self.robot_y + (lx * math.sin(self.robot_yaw) + ly * math.cos(self.robot_yaw))

            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.memory[gy, gx] = min(100, self.memory[gy, gx] + 50)

            angle += msg.angle_increment

        # === Inflate obstacles (safety margin) ===
        self.grid = self.inflate_obstacles(radius_cells=3)

        # Publish occupancy grid
        self.map_pub.publish(self.to_occupancy_grid_msg())

        # === Replan condition ===
        dx = self.robot_x - self.last_replan_pose[0]
        dy = self.robot_y - self.last_replan_pose[1]
        moved = math.hypot(dx, dy) > self.replan_move_thresh

        now = self.get_clock().now()
        time_elapsed = (now - self.last_plan_time).nanoseconds / 1e9

        if obstacle_detected or moved or time_elapsed > 2.0:
            self.start_world = (self.robot_x, self.robot_y)
            self.path_published = False
            self.last_replan_pose = (self.robot_x, self.robot_y)
            self.last_plan_time = now
            if obstacle_detected:
                self.get_logger().warn(
                    f"Obstacle detected near robot. Replanning from ({self.robot_x:.2f}, {self.robot_y:.2f})"
                )

        # === Auto replan if path blocked ===
        if self.path and not self.is_path_safe(self.path):
            self.get_logger().warn("Current path is blocked. Replanning immediately.")
            self.path_published = False

        # === Run planner ===
        if not self.path_published:
            path = self.run_astar()
            if path and self.is_path_safe(path):
                self.publish_path(path)
                self.path = path
                self.path_published = True
                self.get_logger().info(f"New path planned successfully ({len(path)} points).")
            else:
                self.get_logger().warn("No valid path found. Waiting to replan.")

    # ========== LOCAL GOAL COMPUTATION ==========
    def compute_local_goal(self):
        margin = 2 * self.resolution
        half = self.map_size / 2.0 - margin
        lx = np.clip(self.goal_world[0], self.robot_x - half, self.robot_x + half)
        ly = np.clip(self.goal_world[1], self.robot_y - half, self.robot_y + half)
        return (lx, ly)

    # ========== OBSTACLE INFLATION ==========
    def inflate_obstacles(self, radius_cells=2):
        """ขยายสิ่งกีดขวางรอบๆ จุดที่มีค่า occupancy สูง"""
        inflated = np.copy(self.memory)
        for y in range(self.height):
            for x in range(self.width):
                if self.memory[y, x] > 50:
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                inflated[ny, nx] = max(inflated[ny, nx], 80)
        return inflated

    # ========== MESSAGE BUILDERS ==========
    def to_occupancy_grid_msg(self) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        # Dynamic origin (center map around robot)
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.robot_x - self.map_size / 2.0
        msg.info.origin.position.y = self.robot_y - self.map_size / 2.0

        msg.data = self.grid.flatten().astype(np.int8).tolist()
        return msg

    def publish_path(self, path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for (gx, gy) in path:
            wx, wy = self.grid_to_world(gx, gy)
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.get_logger().info("Path published to /astar_path")

    # ========== UTILITIES ==========
    def world_to_grid(self, x, y):
        gx = int((x - (self.robot_x - self.map_size / 2.0)) / self.resolution)
        gy = int((y - (self.robot_y - self.map_size / 2.0)) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = (self.robot_x - self.map_size / 2.0) + gx * self.resolution
        y = (self.robot_y - self.map_size / 2.0) + gy * self.resolution
        return x, y

    # ========== SAFETY ==========
    def is_free(self, node):
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] < 50

    def is_path_safe(self, path):
        for gx, gy in path:
            if not self.is_free((gx, gy)):
                return False
        return True

    # ========== A* PLANNER ==========
    def run_astar(self):
        start = self.world_to_grid(self.robot_x, self.robot_y)
        local_goal_world = self.compute_local_goal()
        goal = self.world_to_grid(*local_goal_world)

        if not self.is_in_bounds(start) or not self.is_in_bounds(goal):
            self.get_logger().warn("Start or local goal is out of map bounds.")
            return None
        if not self.is_free(start) or not self.is_free(goal):
            self.get_logger().warn("Start or local goal position is occupied.")
            return None

        open_set = {start}
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for nb in self.get_neighbors(current):
                if not self.is_free(nb):
                    continue
                tentative = g_score[current] + self.distance(current, nb)
                if tentative < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f_score[nb] = tentative + self.heuristic(nb, goal)
                    open_set.add(nb)
        return None

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def distance(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(self, node):
        x, y = node
        res = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_in_bounds((nx, ny)):
                    res.append((nx, ny))
        return res

    def is_in_bounds(self, node):
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
