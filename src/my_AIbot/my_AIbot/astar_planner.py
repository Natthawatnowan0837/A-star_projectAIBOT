#!/usr/bin/env python3

# เขียนโดย Natthawtnowan (650610837)
# ไฟล์นี้คือ node “AStarPlanner” สำหรับ ROS2
# ทำหน้าที่สร้างเส้นทางให้หุ่นยนต์ โดยใช้ A* algorithm
# และสามารถสลับระหว่าง “เส้นตรงหลัก” กับ “เส้นทางอ้อม” (detour)
# ตามข้อมูล LiDAR ที่ตรวจพบสิ่งกีดขวางได้แบบอัตโนมัติ

import rclpy
from rclpy.node import Node
import numpy as np
import math
from typing import List, Tuple, Optional

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# นิยามชนิดข้อมูลพิกัด
WorldPt = Tuple[float, float]   # พิกัดจริงในโลก (หน่วยเมตร)
GridPt  = Tuple[int, int]       # พิกัดในกริด (cell index)

# ============================================================
#                     MAIN CLASS
# ============================================================
class AStarPlanner(Node):
    """
    วิธีการ
    -----------------
    - หุ่นจะพยายาม “เดินตามเส้นตรงหลัก” จาก start -> goal เป็นหลัก
    - ถ้า LiDAR พบสิ่งกีดขวางบนเส้นทางนั้น → สร้าง “detour path” โดยใช้ A*
    - เมื่ออ้อมผ่านสิ่งกีดขวางได้ → กลับเข้าสู่เส้นตรงหลักอีกครั้ง
    """

    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("AStarPlanner node started successfully.")

        # === QoS สำหรับ LiDAR ===
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # === Subscribers ===
        # รับข้อมูล LiDAR และ Odom
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_scan)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 50)

        # === Publishers ===
        # ส่งข้อมูลแผนที่และเส้นทางให้ RViz ดู
        self.map_pub  = self.create_publisher(OccupancyGrid, '/ocgm', 50)
        self.path_pub = self.create_publisher(Path, '/astar_path', 50)

        # === การตั้งค่าแผนที่ ===
        self.map_size   = 10.0     # แผนที่กว้าง 10 เมตร
        self.resolution = 0.05     # ขนาด cell = 5 ซม.
        self.width      = int(self.map_size / self.resolution)
        self.height     = int(self.map_size / self.resolution)
        self.grid       = np.zeros((self.height, self.width), dtype=np.float32)
        self.memory     = np.zeros_like(self.grid)  # แผนที่ที่มีการ decay ค่า (ลืม obstacle เก่า)

        # === พิกัดเป้าหมาย ===
        self.start_world: WorldPt = (0.0, 0.0)
        self.goal_world:  WorldPt = (4.4, 0.0)

        # === สถานะหุ่นยนต์ ===
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_ready = False
        self.first_odom = True

        # === สถานะเส้นทาง ===
        self.path: List[GridPt] = []
        self.path_published = False
        self.last_plan_time = self.get_clock().now()

        # === เส้นทางตรงหลัก ===
        self.main_path_world: List[WorldPt] = []
        self.main_path_ready = False
        self.last_path_type = 'none'  # 'main' | 'detour' | 'none'

        # === การควบคุมการ replan ===
        self.last_replan_pose: WorldPt = (0.0, 0.0)
        self.replan_move_thresh = 1.0      # ถ้าขยับเกิน 1 ม. ถึงจะ replan ใหม่
        self.min_replan_interval = 5.0     # เวลาขั้นต่ำระหว่าง replan
        self.force_replan_timeout = 10.0   # บังคับ replan ถ้านานเกิน
        self.obstacle_range = 0.35         # ถ้าวัตถุอยู่ใกล้กว่า 35 ซม. ถือว่ามีสิ่งกีดขวาง
        self.rejoin_ahead_steps = 10       # เลือกจุด rejoin ที่อยู่ข้างหน้า 10 จุดบนเส้นหลัก

        # === จำกัดความถี่การ publish ===
        self._last_scan_log = self.get_clock().now()
        self._last_map_pub  = self.get_clock().now()
        self.map_pub_interval = 1.0  # publish map ไม่เกิน 1Hz

    # ============================================================
    #                       CALLBACKS
    # ============================================================

    # ----------------- ODOM -----------------
    def odom_callback(self, msg: Odometry):
        # อ่านตำแหน่งและมุมของหุ่น
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw
        self.odom_ready = True

        # ครั้งแรก: สร้างเส้นตรงหลักจาก start → goal
        if self.first_odom:
            self.start_world = (self.robot_x, self.robot_y)
            self.main_path_world = self.generate_straight_path(self.start_world, self.goal_world, step=self.resolution*2.0)
            self.publish_main_path(self.main_path_world)
            self.main_path_ready = True
            self.last_path_type = 'main'
            self.first_odom = False
            self.get_logger().info(f"Main straight path generated: {len(self.main_path_world)} pts")

    # ----------------- LASER -----------------
    def laser_callback(self, msg: LaserScan):
        if not self.odom_ready:
            return

        # ลด spam log
        now = self.get_clock().now()
        if (now - self._last_scan_log).nanoseconds * 1e-9 > 1.5:
            self.get_logger().info(f"/scan points={len(msg.ranges)}")
            self._last_scan_log = now

        # Decay ค่า memory (ค่อยๆ ลืม obstacle เก่า)
        self.memory = np.clip(self.memory * 0.97, 0, 100)

        # แปลงข้อมูล LiDAR → แผนที่รอบตัวหุ่น
        angle = msg.angle_min
        near_obstacle = False
        for r in msg.ranges:
            if np.isnan(r) or np.isinf(r):
                angle += msg.angle_increment
                continue
            if r < self.obstacle_range:
                near_obstacle = True

            # แปลงจากพิกัด laser → พิกัดโลก (odom)
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            wx = self.robot_x + (lx * math.cos(self.robot_yaw) - ly * math.sin(self.robot_yaw))
            wy = self.robot_y + (lx * math.sin(self.robot_yaw) + ly * math.cos(self.robot_yaw))

            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.memory[gy, gx] = min(100, self.memory[gy, gx] + 80)

            angle += msg.angle_increment

        # ขยายสิ่งกีดขวาง (inflation)
        self.grid = self.inflate_obstacles(radius_cells=4)

        # Publish แผนที่ทุก 1 วินาที
        if (now - self._last_map_pub).nanoseconds * 1e-9 > self.map_pub_interval:
            self.map_pub.publish(self.to_occupancy_grid_msg())
            self._last_map_pub = now

        # =====================================================
        #           REPLANNING LOGIC (main ↔ detour)
        # =====================================================
        time_elapsed = (now - self.last_plan_time).nanoseconds / 1e9
        moved = self.moved_since_last_replan()

        # 1) ถ้ามี obstacle ใกล้เส้นทาง → ทำ detour
        if near_obstacle and (time_elapsed > 1.0):
            if self.distance_to_main_path(self.robot_x, self.robot_y) < 0.5:
                rejoin_goal = self.compute_rejoin_goal()
                path = self.run_astar(goal_world=rejoin_goal)
                if path and self.is_path_safe(path):
                    self.publish_path(path)
                    self.path = path
                    self.path_published = True
                    self.last_path_type = 'detour'
                    self.last_plan_time = now
                    self.last_replan_pose = (self.robot_x, self.robot_y)
                    self.get_logger().warn("Obstacle detected → publish DETOUR path")
                else:
                    self.get_logger().warn("Detour failed (no valid path)")

        # 2) ถ้าไม่เจอ obstacle และอยู่ใกล้เส้นหลัก → กลับไปใช้ main path
        elif not near_obstacle and self.main_path_ready and self.last_path_type != 'main':
            if self.distance_to_main_path(self.robot_x, self.robot_y) < 0.30:
                idx = self.nearest_main_index(self.robot_x, self.robot_y)
                trimmed = self.main_path_world[idx:]
                if len(trimmed) > 2:
                    self.publish_main_path(trimmed)
                    self.last_path_type = 'main'
                    self.last_plan_time = now
                    self.get_logger().info("Rejoined MAIN straight path.")

        # 3) replan ตามเวลา/ระยะทาง เพื่อความปลอดภัย
        if (moved and time_elapsed > self.min_replan_interval) or (time_elapsed > self.force_replan_timeout):
            if self.last_path_type == 'detour':
                rejoin_goal = self.compute_rejoin_goal()
                path = self.run_astar(goal_world=rejoin_goal)
                if path and self.is_path_safe(path):
                    self.publish_path(path)
                    self.path = path
                    self.path_published = True
                    self.last_plan_time = now
                    self.last_replan_pose = (self.robot_x, self.robot_y)
                    self.get_logger().info("Periodic replan for DETOUR path.")

    # ============================================================
    #                 HELPER FUNCTIONS (main path)
    # ============================================================

    def generate_straight_path(self, start: WorldPt, goal: WorldPt, step: float = 0.1) -> List[WorldPt]:
        """สร้างเส้นตรงระหว่าง start กับ goal"""
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        n = max(2, int(dist / max(step, 1e-3)))
        return [(start[0] + t*(goal[0]-start[0]), start[1] + t*(goal[1]-start[1])) for t in np.linspace(0, 1, n)]

    def publish_main_path(self, world_path: List[WorldPt]):
        """ส่งเส้นทางหลักให้ RViz"""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for (wx, wy) in world_path:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.get_logger().info(f"Published MAIN path ({len(world_path)} pts)")

    # ฟังก์ชันหาความใกล้ระหว่างตำแหน่งปัจจุบันกับเส้นหลัก
    def nearest_main_index(self, x, y):
        dists = [(i, (px-x)**2 + (py-y)**2) for i,(px,py) in enumerate(self.main_path_world)]
        return min(dists, key=lambda t: t[1])[0]

    def distance_to_main_path(self, x, y):
        return min(math.hypot(px-x, py-y) for (px,py) in self.main_path_world)

    def compute_rejoin_goal(self) -> WorldPt:
        """หาจุดบน main path ที่อยู่ข้างหน้า เพื่อกลับเข้าเส้น"""
        idx = self.nearest_main_index(self.robot_x, self.robot_y)
        idx2 = min(idx + self.rejoin_ahead_steps, len(self.main_path_world)-1)
        return self.main_path_world[idx2]

    # ============================================================
    #                     A* PLANNER
    # ============================================================
    def run_astar(self, goal_world: Optional[WorldPt] = None) -> Optional[List[GridPt]]:
        """A* algorithm สำหรับหาทางอ้อมสิ่งกีดขวาง"""
        start = self.world_to_grid(self.robot_x, self.robot_y)
        goal = self.world_to_grid(*(goal_world or self.compute_local_goal()))

        if not self.is_in_bounds(start) or not self.is_in_bounds(goal):
            self.get_logger().warn("A*: Start or goal out of bounds.")
            return None
        if not self.is_free(start) or not self.is_free(goal):
            self.get_logger().warn("A*: Start or goal occupied.")
            return None

        # โครงสร้างข้อมูลของ A*
        open_set = {start}
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}
        direction_from = {start: None}

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for nb in self.get_neighbors(current):
                if not self.is_free(nb):
                    continue
                # เพิ่มค่าโทษเมื่อเลี้ยวเยอะ
                prev_dir = direction_from[current]
                new_dir = (nb[0]-current[0], nb[1]-current[1])
                turn_penalty = 0.2 if prev_dir and prev_dir != new_dir else 0.0
                tentative = g_score[current] + self.distance(current, nb) + turn_penalty

                if tentative < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    direction_from[nb] = new_dir
                    g_score[nb] = tentative
                    f_score[nb] = tentative + self.heuristic(nb, goal)
                    open_set.add(nb)
        return None

    # heuristic function แบบระยะตรง
    def heuristic(self, a, b):
        dx, dy = abs(b[0]-a[0]), abs(b[1]-a[1])
        return math.hypot(dx, dy) + 0.05 * abs(dx - dy)

    def reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

# ============================================================
#                     MAIN ENTRY
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
