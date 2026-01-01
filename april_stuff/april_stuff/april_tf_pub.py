#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import time
import numpy as np


class AprilTagStableTF(Node):
    def __init__(self):
        super().__init__('apriltag_stable_tf')

        self.subscription = self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detections',
            self.tag_callback,
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # Store detection history
        self.tag_history = {}  # tag_id -> {pose, start_time, published}

        self.stability_time = 2.0  # seconds
        self.pos_threshold = 0.01  # meters
        self.rot_threshold = 0.02  # quaternion distance

    def tag_callback(self, msg):
        now = time.time()

        for det in msg.detections:
            tag_id = det.id
            pose = det.pose.pose.pose  # geometry_msgs/Pose

            pos = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z])

            quat = np.array([pose.orientation.x,
                             pose.orientation.y,
                             pose.orientation.z,
                             pose.orientation.w])

            if tag_id not in self.tag_history:
                self.tag_history[tag_id] = {
                    'pos': pos,
                    'quat': quat,
                    'start_time': now,
                    'published': False
                }
                continue

            hist = self.tag_history[tag_id]

            pos_diff = np.linalg.norm(pos - hist['pos'])
            quat_diff = np.linalg.norm(quat - hist['quat'])

            if pos_diff < self.pos_threshold and quat_diff < self.rot_threshold:
                if not hist['published'] and (now - hist['start_time']) > self.stability_time:
                    self.publish_tf(tag_id, pose)
                    hist['published'] = True
                    self.get_logger().info(f"Published stable TF for AprilTag {tag_id}")
            else:
                # Reset stability timer
                hist['pos'] = pos
                hist['quat'] = quat
                hist['start_time'] = now

    def publish_tf(self, tag_id, pose):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f'apriltag_{tag_id}'

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        t.transform.rotation = pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = AprilTagStableTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
