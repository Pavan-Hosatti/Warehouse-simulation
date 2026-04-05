import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
    def scan_cb(self, msg): print("GOT SCAN")
    def odom_cb(self, msg): print("GOT ODOM")

rclpy.init()
node = TestNode()
rclpy.spin_once(node, timeout_sec=2.0)
print("Done")
