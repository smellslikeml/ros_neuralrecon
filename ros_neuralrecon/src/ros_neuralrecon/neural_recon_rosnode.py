#!/usr/bin/env python3

import os
import sys
import math
import time

import cv2
import rospy
import rostopic

import numpy as np
from typing import List
from collections import deque

import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Header

from ros_neuralrecon_msgs.msg import SparseTSDF

import torch

from neuralrecon.models import NeuralRecon
from neuralrecon.utils import SaveScene
from transforms3d.quaternions import axangle2quat, quat2mat


def rotate_view_to_align_xyplane(Tr_camera_to_world):
    # world space normal [0, 0, 1]  camera space normal [0, -1, 0]
    z_c = np.dot(np.linalg.inv(Tr_camera_to_world), np.array([0, 0, 1, 0]))[:3]
    axis = np.cross(z_c, np.array([0, -1, 0]))
    axis = axis / np.linalg.norm(axis)
    theta = np.arccos(-z_c[1] / (np.linalg.norm(z_c)))
    quat = axangle2quat(axis, theta)
    rotation_matrix = quat2mat(quat)
    return rotation_matrix


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1))], dim=1)
    xyz_t_h = (transform @ xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(max_depth, size, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    view_frust_pts = torch.stack(
        [
            (torch.tensor([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * torch.tensor([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (torch.tensor([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * torch.tensor([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            torch.tensor([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


class NeuralReconNode:
    """
    This node uses NeuralRecon for tsdf fusion of RGB images and Odometric poses.
    """

    def __init__(self, cfg, rate=10, queue_len=9):
        rostopic._check_master()
        rospy.init_node("neural_recon", anonymous=True)
        rospy.loginfo("%s started" % rospy.get_name())

        self.cfg = cfg
        self.qlen = queue_len
        self.FRAME_HEIGHT, self.FRAME_WIDTH = 300, 300
        self.images = deque(maxlen=self.qlen)
        self.poses = deque(maxlen=self.qlen)

        self.fragment_id = 0
        self.fx = self.FRAME_HEIGHT
        self.fy = self.FRAME_WIDTH
        self.cx = self.FRAME_HEIGHT // 2
        self.cy = self.FRAME_WIDTH // 2

        self.rate = rospy.Rate(rate)
        # self.rate.sleep()

        # TODO: message type to publish/viz output tsdf mesh?
        # self.tsdf = rospy.Publisher(
        #    "tsdf", Twist, queue_size=1
        # )

        self.model = NeuralRecon(self.cfg).cuda().eval()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        saved_models = [
            fn for fn in os.listdir(self.cfg.LOGDIR) if fn.endswith(".ckpt")
        ]
        saved_models = sorted(
            saved_models, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        loadckpt = os.path.join(self.cfg.LOGDIR, saved_models[-1])
        state_dict = torch.load(loadckpt)
        self.epoch_idx = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.save_mesh_scene = SaveScene(self.cfg)

        sync_list = []
        img_sub = message_filters.Subscriber(
            "/rgb_publisher/color/image",
            Image,
        )
        odom_sub = message_filters.Subscriber("/odom", Odometry)
        sync_list.append(img_sub)
        sync_list.append(odom_sub)
        ts = message_filters.ApproximateTimeSynchronizer(sync_list, 1, 1)
        ts.registerCallback(self.callback)

    def callback(self, image, pose):
        img = np.frombuffer(image.data, dtype=np.uint8).reshape(
            self.FRAME_HEIGHT, self.FRAME_WIDTH, -1
        )

        self.images.append(img)
        self.poses.append(pose)

        if len(self.images) == self.qlen:
            img_buffer = []
            imgs = self.images.copy()
            for idx in range(len(imgs)):
                resized_frame = cv2.resize(imgs[idx], (640, 480))
                image_tensor = torch.from_numpy(resized_frame.astype(np.float32))
                image_tensor = torch.transpose(image_tensor, 2, 0)
                img_buffer.append(image_tensor)
            img_tensor = torch.stack(img_buffer)

            cam_dict = {
                "K": np.array(
                    [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
                ),
                "shape": (self.FRAME_WIDTH, self.FRAME_HEIGHT),
                "dist_coeff": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            }
            extrinsics = {}
            intrinsics = {}
            for idx, p in enumerate(self.poses):
                trans = [
                    p.pose.pose.position.x,
                    p.pose.pose.position.y,
                    p.pose.pose.position.z,
                ]
                quat = [
                    p.pose.pose.orientation.x,
                    p.pose.pose.orientation.y,
                    p.pose.pose.orientation.z,
                    p.pose.pose.orientation.w,
                ]
                rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
                trans_mat = np.zeros([3, 4])
                trans_mat[:3, :3] = rot_mat
                trans_mat[:3, 3] = trans
                extrinsics[10 * pose.header.seq + idx] = torch.from_numpy(
                    np.append(trans_mat, np.array([[0, 0, 0, 1]]), axis=0)
                )
                intrinsics[10 * pose.header.seq + idx] = cam_dict

            world_to_aligned = {}
            for idx, p in enumerate(self.poses):
                middle_pose = extrinsics[10 * pose.header.seq + idx].float()
                rotation_matrix = rotate_view_to_align_xyplane(middle_pose)
                rotation_matrix4x4 = np.eye(4)
                rotation_matrix4x4[:3, :3] = rotation_matrix
                world_to_aligned[10 * pose.header.seq + idx] = (
                    torch.from_numpy(rotation_matrix4x4).float() @ middle_pose.inverse()
                )

            proj_matrices = {}
            p_matrices = []
            bnds = torch.zeros((3, 2))
            bnds[:, 0] = np.inf
            bnds[:, 1] = -np.inf
            for idx, p in enumerate(self.poses):
                size = img_tensor[idx].shape[1:]
                view_proj_matrices = []
                ints = torch.from_numpy(
                    intrinsics[10 * pose.header.seq + idx]["K"]
                ).float()
                exts = extrinsics[10 * pose.header.seq + idx].float()
                view_frust_pts = get_view_frustum(3.0, size, ints, exts)
                bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
                bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])
                for i in range(3):
                    proj_mat = torch.inverse(exts.data.cpu())
                    scale_intrinsics = ints / 2**i
                    scale_intrinsics[-1, -1] = 1
                    proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4].float()
                    view_proj_matrices.append(proj_mat)
                p_matrices.append(torch.stack(view_proj_matrices))
            p_matrices = torch.stack(p_matrices)
            proj_matrices[10 * pose.header.seq + idx] = torch.unsqueeze(p_matrices, 0)

            # -------adjust volume bounds-------
            num_layers = 3
            center = (
                torch.tensor(
                    ((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2, -0.2)
                )
                - torch.from_numpy(np.array([0, 0, 0]))
            ) / cfg.MODEL.VOXEL_SIZE
            center[:2] = torch.round(center[:2] / 2**num_layers) * 2**num_layers
            center[2] = torch.floor(center[2] / 2**num_layers) * 2**num_layers
            origin = torch.zeros_like(center)
            origin[:2] = center[:2] - torch.tensor(cfg.MODEL.N_VOX[:2]) // 2
            origin[2] = center[2]
            vol_origin_partial = origin * cfg.MODEL.VOXEL_SIZE + torch.from_numpy(
                np.array([0, 0, 0])
            )

            #################

            sample = {
                "imgs": img_tensor[None, :, :, :, :],
                "intrinsics": torch.from_numpy(np.stack(intrinsics)),
                "extrinsics": torch.from_numpy(np.stack(extrinsics)),
                "proj_matrices": proj_matrices[10 * pose.header.seq + idx],
                "world_to_aligned_camera": torch.unsqueeze(
                    world_to_aligned[10 * pose.header.seq + idx], 0
                ),
                "scene": ["ignore"],
                "fragment": ["ignore" + "_" + str(self.fragment_id)],
                "epoch": torch.from_numpy(np.array([0])),
                "vol_origin": torch.from_numpy(np.array([0, 0, 0])),
                "vol_origin_partial": vol_origin_partial,  # torch.from_numpy(np.array([0, 0, 0])),
            }

            self.fragment_id += 1
            # Generate Sample Dict & Run NeuralRecon
            with torch.no_grad():
                outputs, loss_dict = self.model(sample)
                if self.cfg.SAVE_SCENE_MESH or self.cfg.SAVE_INCREMENTAL:
                    self.save_mesh_scene(outputs, sample, self.epoch_idx)
                if "coords" in outputs.keys():
                    rospy.loginfo(str(outputs["coords"][0]))
                    rospy.loginfo(str(outputs["tsdf"][0]))

            rospy.loginfo("updated scene")

            self.images = deque(maxlen=self.qlen)
            self.poses = deque(maxlen=self.qlen)

        self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("Shutting down neural recon")
        self.save_mesh_scene.close()
        rospy.signal_shutdown("Shutdown all threads")
        sys.exit(0)


if __name__ == "__main__":
    import os, rospkg
    from neuralrecon.config import cfg, update_config

    rp = rospkg.RosPack()
    cfg.LOGDIR = os.path.join(rp.get_path("ros_neuralrecon"), "checkpoints")
    cfg.SAVE_INCREMENTAL = True
    cfg.MODEL.BACKBONE2D.ARC = rospy.get_param("/ros_neuralrecon/arc")
    cfg.MODEL.FUSION.FUSION_ON = bool(rospy.get_param("/ros_neuralrecon/fusion_on"))

    ros_node = NeuralReconNode(cfg=cfg)
    rospy.on_shutdown(ros_node.shutdown)
    rospy.spin()
