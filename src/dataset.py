import h5py
import numpy as np
import os
from numpy.lib.shape_base import expand_dims
import open3d as o3d
import torch
from torch.utils.data import Dataset

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class ModelNet40(Dataset):
    """docstring for ModelNet40"""
    def __init__(self, prefix, args):
        self.prefix = prefix
        self.gaussian_noise = args.gaussian_noise
        self.unseen = args.unseen
        self.partial = args.partial

        # self.filtering = args.filtering
        
        self.n_cpoints = 1024
        self.n_ppoints = 768
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        # self.use_ppf = args.use_ppf
        # self.use_fpfh = args.use_fpfh

        if self.prefix == "train":
            path = './data/ModelNet40_train.h5'
        elif self.prefix == "val":
            if args.small_angle:
                path = './data/ModelNet40_test_SmallAngle.h5'
            else:
                path = './data/ModelNet40_test.h5'

        f = h5py.File(path, 'r')
        self.data = np.array(f['data'][:].astype('float32'))
        self.label = np.squeeze(np.array(f['label'][:].astype('int64')))
        print(self.data.shape, self.label.shape)

        if self.prefix == "val":
            self.src_cc = np.array(f['complete_src_clean'][:].astype('float32'))
            self.tgt_cc = np.array(f['complete_tgt_clean'][:].astype('float32'))
            self.src_cn = np.array(f['complete_src_noise'][:].astype('float32'))
            self.tgt_cn = np.array(f['complete_tgt_noise'][:].astype('float32'))

            self.src_pc = np.array(f['partial_src_clean'][:].astype('float32'))
            self.tgt_pc = np.array(f['partial_tgt_clean'][:].astype('float32'))
            self.src_pn = np.array(f['partial_src_noise'][:].astype('float32'))
            self.tgt_pn = np.array(f['partial_tgt_noise'][:].astype('float32'))

            self.transforms = np.array(f['transforms'][:].astype('float32'))
        f.close()

        if self.unseen:
            if self.prefix == "val":
                self.data = self.data[self.label>=20]

                self.src_cc = self.src_cc[self.label>=20]
                self.tgt_cc = self.tgt_cc[self.label>=20]
                self.src_cn = self.src_cn[self.label>=20]
                self.tgt_cn = self.tgt_cn[self.label>=20]

                self.src_pc = self.src_pc[self.label>=20]
                self.tgt_pc = self.tgt_pc[self.label>=20]
                self.src_pn = self.src_pn[self.label>=20]
                self.tgt_pn = self.tgt_pn[self.label>=20]

                self.transforms = self.transforms[self.label>=20]

                self.label = self.label[self.label>=20]
            elif self.prefix == "train":
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

        print(self.data.shape, self.label.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.prefix == "train":
            src = self.data[index][:self.n_cpoints]
            tgt = src
            
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1

            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]

            src = np.random.permutation(src)
            tgt = np.random.permutation(tgt)

            if self.partial:
                src, tgt = farthest_subsample_points(src.T, tgt.T, num_subsampled_points=self.n_ppoints)
                src = src.T
                tgt = tgt.T

            if self.gaussian_noise:
                src = jitter_pcd(src)
                tgt = jitter_pcd(tgt)

        elif self.prefix == "val":
            if self.partial:
                if self.gaussian_noise:
                    src = self.src_pn[index][:self.n_ppoints]
                    tgt = self.tgt_pn[index][:self.n_ppoints]
                else:
                    src = self.src_pc[index][:self.n_ppoints]
                    tgt = self.tgt_pc[index][:self.n_ppoints]
            else:
                if self.gaussian_noise:
                    src = self.src_cn[index][:self.n_cpoints]
                    tgt = self.tgt_cn[index][:self.n_cpoints]
                else:
                    src = self.src_cc[index][:self.n_cpoints]
                    tgt = self.tgt_cc[index][:self.n_cpoints]
            transform = self.transforms[index]

            src = np.random.permutation(src)
            tgt = np.random.permutation(tgt)
        
        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)

        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            match_level = 1
            rot_level = 1
            return src, tgt, transform, match_level, rot_level
        else:
            return src, tgt