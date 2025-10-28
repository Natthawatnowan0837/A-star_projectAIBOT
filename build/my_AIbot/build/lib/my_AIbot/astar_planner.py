#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


WorldPt = Tuple[float, float]
GridPt  = Tuple[int, int]


class AStarPlanner(Node):
    """
    กลยุทธ์: ให้หุ่น "เดินตามเส้นตรงหลัก" (main straight path) ไปยัง goal เสมอ
    - ถ้า LiDAR พบสิ่งกีดขวางใกล้เส้นทางหลัก -> ใช้ A* สร้าง 'detour path' อ้อมไปยัง 'จุดกลับเข้าเส้นตรง'
    - เมื่อพ้นสิ่งกีดขวาง/อยู่ใกล้เส้นหลักอีกครั้ง -> publish 'ส่วนที่เหลือของเส้นตรงหลัก' ต่อจนถึง goal

    หมายเหตุ:
    - แผนที่เป็น robot-centric (origin เลื่อนตามหุ่น) ดังนั้นใช้ frame_id = 'odom' ทั้ง OccupancyGrid/Path
    - จำกัดความถี่ publish เพื่อลด "Message Filter queue full" ใน RViz
    """

    def __init__(self):
        super().__init__('astar_planner')

        # === QoS ===
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # === Subscribers ===
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_scan)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 50)

        # === Publishers ===
        self.map_pub  = self.create_publisher(OccupancyGrid, '/ocgm', 50)
        self.path_pub = self.create_publisher(Path, '/astar_path', 50)

        # === Map parameters ===
        self.map_size   = 10.0     # meters (edge length)
        self.resolution = 0.05     # m/cell
        self.width      = int(self.map_size / self.resolution)
        self.height     = int(self.map_size / self.resolution)
        self.grid       = np.zeros((self.height, self.width), dtype=np.float32)  # int8 จะสร้างตอน publish
        self.memory     = np.zeros_like(self.grid)  # float map with decay

        # === World goal ===
        self.start_world: WorldPt = (0.0, 0.0)
        self.goal_world:  WorldPt = (4.0, 0.0)

        # === Robot state ===
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_ready = False
        self.first_odom = True

        # === Path state ===
        self.path: List[GridPt] = []
        self.path_published = False
        self.last_plan_time = self.get_clock().now()

        # === Main straight path ===
        self.main_path_world: List[WorldPt] = []
        self.main_path_ready = False
        self.last_path_type = 'none'  # 'main' | 'detour' | 'none'

        # === Replan controls ===
        self.last_replan_pose: WorldPt = (0.0, 0.0)
        self.replan_move_thresh = 1.0        # m ต้องขยับเกินค่านี้ถึงจะอนุญาต replan จาก movement
        self.min_replan_interval = 5.0       # s คั่นระหว่าง replan
        self.force_replan_timeout = 10.0     # s บังคับ replan หากนานเกิน
        self.obstacle_range = 0.35           # m ระยะที่ถือว่ามีสิ่งกีดขวางใกล้
        self.rejoin_ahead_steps = 10         # ก้าวล่วงหน้าบน main path เมื่อเลือกจุดกลับเข้าเส้น

        # === Publishing throttle ===
        self._last_scan_log = self.get_clock().now()
        self._last_map_pub  = self.get_clock().now()
        self.map_pub_interval = 1.0  # s publish /ocgm ไม่เกิน ~1Hz

        self.get_logger().info("AStarPlanner node started successfully.")

    # =====================  ODOM  =====================
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw
        self.odom_ready = True

        # ตั้ง start_world จาก odom ครั้งแรกให้ตรงกับตำแหน่งเริ่มจริง
        if self.first_odom:
            self.start_world = (self.robot_x, self.robot_y)
            # สร้างเส้นทางตรงหลักจาก start -> goal หนึ่งครั้ง
            self.main_path_world = self.generate_straight_path(self.start_world, self.goal_world, step=self.resolution*2.0)
            self.publish_main_path(self.main_path_world)  # publish main ทันที
            self.main_path_ready = True
            self.last_path_type = 'main'
            self.first_odom = False
            self.get_logger().info(f"Main straight path generated: {len(self.main_path_world)} pts")

    # ================  LASER / MAPPING  ================
    def laser_callback(self, msg: LaserScan):
        if not self.odom_ready:
            return

        # ลด log spam จาก /scan
        now = self.get_clock().now()
        if (now - self._last_scan_log).nanoseconds * 1e-9 > 1.5:
            self.get_logger().info(f"/scan points={len(msg.ranges)} rmin={msg.range_min:.2f} rmax={msg.range_max:.2f}")
            self._last_scan_log = now

        # Decay memory map
        self.memory = np.clip(self.memory * 0.97, 0, 100)

        # Scan → memory
        angle = msg.angle_min
        near_obstacle = False
        for r in msg.ranges:
            if np.isnan(r) or np.isinf(r) or r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            if r < self.obstacle_range:
                near_obstacle = True

            # LiDAR (laser frame) → world(odom) โดยใช้ทิศหุ่น
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            wx = self.robot_x + (lx * math.cos(self.robot_yaw) - ly * math.sin(self.robot_yaw))
            wy = self.robot_y + (lx * math.sin(self.robot_yaw) + ly * math.cos(self.robot_yaw))

            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                # เพิ่มความเข้มของ obstacle
                self.memory[gy, gx] = min(100, self.memory[gy, gx] + 80)

            angle += msg.angle_increment

        # Inflate obstacles (safety margin)
        self.grid = self.inflate_obstacles(radius_cells=4)

        # Publish map (throttled ~1Hz)
        if (now - self._last_map_pub).nanoseconds * 1e-9 > self.map_pub_interval:
            self.map_pub.publish(self.to_occupancy_grid_msg())
            self._last_map_pub = now

        # ==== Replanning Policy (Hybrid) ====
        # 1) ถ้าพบ obstacle ใกล้ "และ" อยู่ใกล้เส้นตรงหลัก → สร้าง detour A*
        # 2) ถ้าไม่มี obstacle และเราอยู่ใกล้เส้นหลัก → กลับไปใช้ main path ส่วนที่เหลือ
        time_elapsed = (now - self.last_plan_time).nanoseconds / 1e9
        moved = self.moved_since_last_replan()

        # 1) ทำ detour เมื่อจำเป็น
        if near_obstacle and (time_elapsed > 1.0):  # กันสั่นเล็กน้อย
            if self.distance_to_main_path(self.robot_x, self.robot_y) < 0.5:
                rejoin_goal = self.compute_rejoin_goal()
                path = self.run_astar(goal_world=rejoin_goal)
                if path and self.is_path_safe(path):
                    self.publish_path(path)           # detour path
                    self.path = path
                    self.path_published = True
                    self.last_path_type = 'detour'
                    self.last_plan_time = now
                    self.last_replan_pose = (self.robot_x, self.robot_y)
                    self.get_logger().warn("Obstacle on main path → publish DETOUR path")
                else:
                    self.get_logger().warn("Detour failed (no valid A* path).")

        # 2) ถ้าไม่เจอ obstacle → กลับ main path (เฉพาะเมื่อเคย detour อยู่)
        elif not near_obstacle and self.main_path_ready and self.last_path_type != 'main':
            if self.distance_to_main_path(self.robot_x, self.robot_y) < 0.30:
                # publish main path ส่วนที่เหลือ (trim จากจุดใกล้สุด)
                idx = self.nearest_main_index(self.robot_x, self.robot_y)
                trimmed = self.main_path_world[idx:]
                if len(trimmed) > 2:
                    self.publish_main_path(trimmed)
                    self.last_path_type = 'main'
                    self.last_plan_time = now
                    self.get_logger().info("Rejoined MAIN straight path.")

        # 3) เงื่อนไข replan ปกติ (กัน path ตันหรือเงื่อนไขเวลา/ระยะ)
        if (moved and time_elapsed > self.min_replan_interval) or (time_elapsed > self.force_replan_timeout):
            # replan เล็กน้อยเพื่ออัปเดตความปลอดภัยของ path ปัจจุบัน (ถ้ากำลังเดิน main อยู่ ก็ไม่ต้อง publish ซ้ำ)
            if self.last_path_type == 'detour':
                rejoin_goal = self.compute_rejoin_goal()
                path = self.run_astar(goal_world=rejoin_goal)
                if path and self.is_path_safe(path):
                    self.publish_path(path)
                    self.path = path
                    self.path_published = True
                    self.last_plan_time = now
                    self.last_replan_pose = (self.robot_x, self.robot_y)
                    self.get_logger().info("Periodic replan for DETOUR (maintenance).")

    # ================== MAIN PATH HELPERS ==================
    def generate_straight_path(self, start: WorldPt, goal: WorldPt, step: float = 0.1) -> List[WorldPt]:
        """สร้างเส้นตรงระหว่าง start กับ goal ในพิกัดโลก (odom)"""
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        n = max(2, int(dist / max(step, 1e-3)))
        path: List[WorldPt] = []
        for i in range(n + 1):
            t = i / n
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))
        return path

    def publish_main_path(self, world_path: List[WorldPt]):
        """Publish เส้นทางตรงหลักใน frame 'odom'"""
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
        self.get_logger().info(f"Published MAIN straight path ({len(world_path)} pts)")

    def nearest_main_index(self, x: float, y: float) -> int:
        if not self.main_path_world:
            return 0
        dists = [ (i, (px - x)*(px - x) + (py - y)*(py - y)) for i,(px,py) in enumerate(self.main_path_world) ]
        return min(dists, key=lambda t: t[1])[0]

    def distance_to_main_path(self, x: float, y: float) -> float:
        if not self.main_path_world:
            return float('inf')
        return min(math.hypot(px - x, py - y) for (px, py) in self.main_path_world)

    def compute_rejoin_goal(self) -> WorldPt:
        """หาจุดบน main path ที่อยู่ข้างหน้าจากจุดใกล้สุด เพื่อกลับเข้าเส้นหลัก"""
        if not self.main_path_world:
            return self.goal_world
        idx = self.nearest_main_index(self.robot_x, self.robot_y)
        idx2 = min(idx + self.rejoin_ahead_steps, len(self.main_path_world) - 1)
        return self.main_path_world[idx2]

    # =================== MAP / MESSAGES ===================
    def to_occupancy_grid_msg(self) -> OccupancyGrid:
        """สร้าง OccupancyGrid แบบ robot-centric (origin เลื่อนตามหุ่น) → ใช้ frame 'odom'"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'  # สำคัญ: ให้สอดคล้องกับ origin ที่เลื่อนตามหุ่น

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        # origin = มุมซ้ายล่างของกริดในโลก (odom)
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.robot_x - self.map_size / 2.0
        msg.info.origin.position.y = self.robot_y - self.map_size / 2.0

        # แปลง memory -> int8 (0=free, 100=occupied)
        grid_int = np.zeros_like(self.memory, dtype=np.int8)
        grid_int[self.memory >= 50] = 100
        # ถ้าต้องการแสดงพื้นที่ "ไม่แน่ใจ" สามารถตั้งค่า 30 ให้ memory ระหว่าง 1..49 ได้
        # grid_int[(self.memory > 0) & (self.memory < 50)] = 30

        msg.data = grid_int.flatten().tolist()
        return msg

    def publish_path(self, path: List[GridPt]):
        """Publish path (grid) เป็น Path ใน frame 'odom'"""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for (gx, gy) in path:
            wx, wy = self.grid_to_world(gx, gy)
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.get_logger().info(f"Published DETOUR path ({len(path)} pts)")

    # ===================== UTILITIES =====================
    def world_to_grid(self, x: float, y: float) -> GridPt:
        gx = int((x - (self.robot_x - self.map_size / 2.0)) / self.resolution)
        gy = int((y - (self.robot_y - self.map_size / 2.0)) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> WorldPt:
        x = (self.robot_x - self.map_size / 2.0) + gx * self.resolution
        y = (self.robot_y - self.map_size / 2.0) + gy * self.resolution
        return x, y

    def moved_since_last_replan(self) -> bool:
        dx = self.robot_x - self.last_replan_pose[0]
        dy = self.robot_y - self.last_replan_pose[1]
        return math.hypot(dx, dy) > self.replan_move_thresh

    # ================== SAFETY / FREE CELL ==================
    def is_free(self, node: GridPt) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] < 50

    def is_path_safe(self, path: List[GridPt]) -> bool:
        for gx, gy in path:
            if not self.is_free((gx, gy)):
                return False
        return True

    # ==================== OBSTACLE INFLATION ====================
    def inflate_obstacles(self, radius_cells: int = 4) -> np.ndarray:
        """ขยายสิ่งกีดขวางรอบจุดที่มีค่า memory สูง เพื่อ margin ความปลอดภัย"""
        inflated = np.copy(self.memory)
        # วงกลม (disk) mask
        rr = radius_cells
        ys, xs = np.ogrid[-rr:rr+1, -rr:rr+1]
        mask = xs*xs + ys*ys <= rr*rr

        # หา index ที่เป็น obstacle
        ys_idx, xs_idx = np.where(self.memory > 50)
        for (y0, x0) in zip(ys_idx, xs_idx):
            y1 = max(0, y0 - rr); y2 = min(self.height, y0 + rr + 1)
            x1 = max(0, x0 - rr); x2 = min(self.width,  x0 + rr + 1)
            sub = inflated[y1:y2, x1:x2]
            my = mask[(y1 - (y0 - rr)):(y2 - (y0 - rr)), (x1 - (x0 - rr)):(x2 - (x0 - rr))]
            sub[my] = np.maximum(sub[my], 80)
        return inflated

    # ======================= A* PLANNER ========================
    def run_astar(self, goal_world: Optional[WorldPt] = None) -> Optional[List[GridPt]]:
        """A* จากตำแหน่งหุ่น → goal_world (หรือ local goal) บนกริด robot-centric
           - มี directional smoothing เพื่อลดการซิกแซก
        """
        start = self.world_to_grid(self.robot_x, self.robot_y)
        if goal_world is None:
            local_goal_world = self.compute_local_goal()
        else:
            local_goal_world = goal_world
        goal = self.world_to_grid(*local_goal_world)

        if not self.is_in_bounds(start) or not self.is_in_bounds(goal):
            self.get_logger().warn("A*: Start or goal out of bounds.")
            return None
        if not self.is_free(start) or not self.is_free(goal):
            self.get_logger().warn("A*: Start or goal occupied.")
            return None

        open_set = {start}
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}
        direction_from = {start: None}   # ทิศทางก่อนหน้าเพื่อลด turn

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for nb in self.get_neighbors(current):
                if not self.is_free(nb):
                    continue

                # Directional smoothing: ลดการหักเลี้ยวโดยไม่จำเป็น
                prev_dir = direction_from[current]
                new_dir = (nb[0] - current[0], nb[1] - current[1])
                turn_penalty = 0.0
                if prev_dir is not None and prev_dir != new_dir:
                    turn_penalty = 0.2  # ปรับได้ 0.1-0.3

                tentative = g_score[current] + self.distance(current, nb) + turn_penalty

                if tentative < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    direction_from[nb] = new_dir
                    g_score[nb] = tentative
                    f_score[nb] = tentative + self.heuristic(nb, goal)
                    open_set.add(nb)
        return None

    def compute_local_goal(self) -> WorldPt:
        """จำกัด goal ให้อยู่ในกรอบแผนที่เลื่อนตามหุ่น (เผื่อ goal ไกลเกิน map)"""
        margin = 2 * self.resolution
        half = self.map_size / 2.0 - margin
        lx = np.clip(self.goal_world[0], self.robot_x - half, self.robot_x + half)
        ly = np.clip(self.goal_world[1], self.robot_y - half, self.robot_y + half)
        return (lx, ly)

    def heuristic(self, a: GridPt, b: GridPt) -> float:
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        # straight-line bias + เล็กน้อยสำหรับความต่างแกน เพื่อลดการสลับแกนบ่อย
        return math.hypot(dx, dy) + 0.05 * abs(dx - dy)

    def distance(self, a: GridPt, b: GridPt) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(self, node: GridPt) -> List[GridPt]:
        x, y = node
        res: List[GridPt] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not self.is_in_bounds((nx, ny)):
                    continue
                # ลดโอกาสเลือกทางเฉียง (ให้เส้นตรงเด่นขึ้น)
                if abs(dx) + abs(dy) == 2:
                    # อนุญาตเฉียงเมื่อไม่บาดกำแพง (no corner cutting)
                    if self.is_in_bounds((x + dx, y)) and self.is_in_bounds((x, y + dy)):
                        if self.is_free((x + dx, y)) and self.is_free((x, y + dy)):
                            res.append((nx, ny))
                else:
                    res.append((nx, ny))
        return res

    def is_in_bounds(self, node: GridPt) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def reconstruct_path(self, came_from: dict, cur: GridPt) -> List[GridPt]:
        path: List[GridPt] = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        # (optional) smoothing เพิ่มเติมสามารถใส่ได้ที่นี่ถ้าต้องการ
        return path


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
