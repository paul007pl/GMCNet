import sys
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

from scipy.spatial import cKDTree
from pycuda import gpuarray
from pycuda.compiler import SourceModule

from copy import deepcopy


_EPS = 1e-5  # To prevent division by zero


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    scores = (torch.matmul(query.cpu(), key.transpose(-2, -1).contiguous().cpu()) / math.sqrt(d_k)).cuda()
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2, dim=0, keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def get_edge_features(x, idx):
    batch_size, num_points, k = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous() 
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # B N k C
    return feature


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    _, num_dims, _ = x.size()
    feature = get_edge_features(x, idx)
    x = x.transpose(2, 1).contiguous()
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knn_point(k, point_input, point_output=None):
    """
        k: int
        point_input: B n C
        point_output: B m C
    """
    if point_output == None:
        point_output = point_input

    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    idx = pairwise_distance.topk(k=k, dim=-1)[1].detach() # (batch_size, m, k)
    return idx


def knn_idx(pts, k):
    kdt = cKDTree(pts) 
    _, idx = kdt.query(pts, k=k+1)
    return idx[:, 1:]


def get_rri_cuda(pts, k, npts_per_block=1):
    import pycuda.autoinit
    mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
    rri_cuda = mod_rri.get_function('get_rri_feature')

    N = len(pts)
    pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
    k_idx = knn_idx(pts, k)
    k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
    feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

    rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
             grid=(((N-1) // npts_per_block)+1, 1),
             block=(npts_per_block, k, 1))
    
    feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
    return feat

def get_rri_cluster_cuda(pts, k, npts_per_block=1):
    xyz = pts.cpu().numpy() # B N 3
    res = np.zeros((xyz.shape[0], xyz.shape[1], 4*k))

    for i in range(len(xyz)):
        cur_pts = xyz[i]
        feat = get_rri_cuda(cur_pts, k, npts_per_block)
        res[i] = np.array(feat)
    
    res = torch.from_numpy(res).float().cuda() # B N 4k
    return res


def get_rri_cluster(cluster_pts, k):
    '''
    Input:
        cluster_pts: B 3 S M; k: int;
    Output:
        cluster_feats: B 4k S M;
    '''
    batch_size = cluster_pts.size()[0]
    num_samples = cluster_pts.size()[2]
    num_clusters = cluster_pts.size()[3]

    cluster_pts_ = cluster_pts.permute(0, 3, 1, 2).contiguous().view(batch_size*num_clusters, 3, num_samples) # BM 3 S
    idx = knn(cluster_pts_, k+1)[:,:,1:] # BM S k
    cluster_npts_ = (get_edge_features(cluster_pts_, idx)).permute(0, 3, 2, 1).contiguous() # BM 3 k S

    p = cluster_pts_.transpose(1, 2).contiguous().unsqueeze(2).repeat(1,1,k,1) # BM S k 3
    q = cluster_npts_.transpose(1, 3).contiguous() # BM S k 3

    rp = torch.norm(p, None, dim=-1, keepdim=True) # BM S k 1
    rq = torch.norm(q, None, dim=-1, keepdim=True) # BM S k 1
    pn = p / rp
    qn = q / rq
    dot = torch.sum(pn * qn, dim=-1, keepdim=True) # BM S k 1

    theta = torch.acos(torch.clamp(dot, -1, 1)) # BM S k 1

    # T_q = q - dot * p # BM S k 3
    # # BM S k k
    # sin_psi = torch.sum(torch.cross(T_q[:,:,None].repeat(1,1,k,1,1), T_q[:,:,:,None].repeat(1,1,1,k,1), dim=-1) * pn[:,:,None], -1)
    # cos_psi = torch.sum(T_q[:,:,None] * T_q[:,:,:,None], -1)
    # psi = torch.atan2(sin_psi, cos_psi) % (2*np.pi)
    # # psi = torch.from_numpy(np.arctan2(sin_psi.cpu().numpy(), cos_psi.cpu().numpy()) % (2*np.pi)).cuda()
    # # BM S k 1
    # _, idx = psi.topk(k=2, dim=-1, largest=False)
    # idx = idx[:, :, :, 1:2]
    # phi = torch.gather(psi, dim=-1, index=idx)

    T_q = (q - dot * p)
    T_q = T_q.cpu().numpy()
    pn = pn.cpu().numpy()
    sin_psi = np.sum(np.cross(T_q[:, :, None], T_q[:, :, :, None]) * pn[:, :, None], -1)
    cos_psi = np.sum(T_q[:, :, None] * T_q[:, :, :, None], -1)

    psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)

    # psi = np.where(psi < 0, psi+2*np.pi, psi)

    idx = np.argpartition(psi, 1)[:, :, :, 1:2]
    # phi: BM x S x k x 1, projection angles
    phi = torch.from_numpy(np.take_along_axis(psi, idx, axis=-1)).to(theta.device)

    feat = torch.cat([rp, rq, theta, phi], axis=-1).view(batch_size, num_clusters, num_samples, 4*k).transpose(1,3).contiguous() # B 4k S M
    return feat


def batch_choice(data, k, p=None, replace=False):
    # data is [B, N]
    out = []
    for i in range(len(data)):
        out.append(np.random.choice(data[i], size=k, p=p[i], replace=replace))
    out = np.stack(out, 0)
    return out


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=1):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def estimate_fpfh(pts, nms, radius=0.1, max_nn=30):
    pts = pts.cpu().numpy()
    nms = nms.cpu().numpy()
    feats = np.zeros((pts.shape[0], 33, pts.shape[1]))
    for i in range(len(pts)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[i])
        pcd.normals = o3d.utility.Vector3dVector(nms[i])
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        feats[i] = pcd_fpfh.data
    feats = torch.from_numpy(feats).float().cuda()
    return feats

def angle(v1, v2):
    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                                v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                                v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def sinkhorn(log_alpha, num_iters: int = 5):
    zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

    log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

    for i in range(num_iters):
        # Row normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)
    return log_alpha_padded


def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.transpose(-1, -2)) # B * N M
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1) # B * N 1
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(-2) # B * 1 M
    return dist


def get_cluster_feats(feats, pts, k=10, return_cluster_pts=False):
    idx = knn(pts.transpose(1,2).contiguous(), k) # B N k
    batch_size, num_dims, num_points = feats.size()

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    feats = feats.transpose(2, 1).contiguous() # B, N, C
    out_feats = feats.view(batch_size * num_points, -1)[idx, :]
    out_feats = out_feats.view(batch_size, num_points, k, num_dims).permute(0, 3, 1, 2).contiguous() # B C N k

    if return_cluster_pts:
        out_pts = pts.view(batch_size * num_points, -1)[idx, :]
        out_pts = out_pts.view(batch_size, num_points, k, 3).permute(0, 3, 1, 2).contiguous() # B 3 N k
        return out_feats, out_pts
    else:
        return out_feats


def sample_and_group_feats(num_centers, num_samples, pts, nms, feats=None, rif="ppf"):
    B, N, C = pts.shape

    if num_centers > 0:
        M = num_centers
        fps_idx = furthest_point_sample(pts, num_centers)
        cent_pts = gather_points(pts.transpose(1,2).contiguous(), fps_idx).transpose(1,2).contiguous() # B M 3
        pn_idx = knn_point(num_samples, pts, cent_pts).detach().int() # B M S
        sm_pts = grouping_operation(pts.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous() # B M S 3

        if rif == "ppf" or rif == "fpfh":
            cent_nms = gather_points(nms.transpose(1,2).contiguous(), fps_idx).transpose(1,2).contiguous()
            sm_nms = grouping_operation(nms.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous() # B M S 3
        else:
            cent_nms = None
            sm_nms = None
        if feats is not None:
            sm_feats = grouping_operation(feats, pn_idx) # B C M S
        else:
            sm_feats = None
    else:
        M = N
        # fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1).to(xyz.device)
        cent_pts = pts # B N 3
        cent_nms = nms # B N 3
        
        if rif == "ppf" or rif == "fpfh":
            if feats is not None:
                all_feats = torch.cat([cent_pts.transpose(1,2).contiguous(), cent_nms.transpose(1,2).contiguous(), feats], dim=1) # B C M
                sm_all_feats = get_cluster_feats(all_feats, cent_pts, num_samples) # B C M S
                sm_pts = (sm_all_feats[:, 0:3, :, :]).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_nms = (sm_all_feats[:, 3:6, :, :]).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_feats = sm_all_feats[:, 6:, :, :]
            else:
                all_feats = torch.cat([cent_pts.transpose(1,2).contiguous(), cent_nms.transpose(1,2).contiguous()], dim=1) # B 6 M
                sm_all_feats = get_cluster_feats(all_feats, cent_pts, num_samples) # B C M S
                sm_pts = (sm_all_feats[:, 0:3, :, :]).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_nms = (sm_all_feats[:, 3:6, :, :]).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_feats = None
        elif rif == "rri":
            if feats is not None:
                all_feats = torch.cat([cent_pts.transpose(1,2).contiguous(), feats], dim=1) # B C M
                sm_all_feats = get_cluster_feats(all_feats, cent_pts, num_samples) # B C M S
                sm_pts = (sm_all_feats[:, 0:3, :, :]).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_feats = sm_all_feats[:, 3:, :, :]
            else:
                sm_pts = get_cluster_feats(cent_pts.transpose(1,2).contiguous(), cent_pts, num_samples).permute(0, 2, 3, 1).contiguous() # B M S 3
                sm_feats = None
            cent_nms = None
    
    d = sm_pts - cent_pts.unsqueeze(2) # B M S 3
    xyz_feats = d  # B M S 3
    
    if rif == "ppf" or rif == "fpfh":
        nr_d = angle(cent_nms.unsqueeze(2), d) 
        ni_d = angle(sm_nms, d)
        nr_ni = angle(cent_nms.unsqueeze(2), sm_nms)
        d_norm = torch.norm(d, dim=-1)

        rif_feats = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # B M S 4
    elif rif == "rri":
        cluster_pts_ = sm_pts.view(B*M, num_samples, 3).transpose(1, 2).contiguous() # BM 3 S
        idx = knn(cluster_pts_, 2+1)[:,:,1:]
        cluster_npts_ = (get_edge_features(cluster_pts_, idx)).permute(0, 3, 2, 1).contiguous() # BM 3 2 S
        p = cluster_pts_.transpose(1, 2).contiguous().unsqueeze(2).repeat(1,1,2,1) # BM S k 3
        q = cluster_npts_.transpose(1, 3).contiguous() # BM S 2 3

        rp = torch.norm(p, None, dim=-1, keepdim=True) # BM S 2 1
        rq = torch.norm(q, None, dim=-1, keepdim=True) # BM S 2 1
        pn = p / rp
        qn = q / rq
        dot = torch.sum(pn * qn, dim=-1, keepdim=True) # BM S 2 1

        theta = torch.acos(torch.clamp(dot, -1, 1)) # BM S 2 1

        T_q = (q - dot * p)
        T_q = T_q.cpu().numpy()
        pn = pn.cpu().numpy()
        sin_psi = np.sum(np.cross(T_q[:, :, None], T_q[:, :, :, None]) * pn[:, :, None], -1)
        cos_psi = np.sum(T_q[:, :, None] * T_q[:, :, :, None], -1)

        psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
        idx = np.argpartition(psi, 1)[:, :, :, 1:2]
        # phi: BM x S x k x 1, projection angles
        phi = torch.from_numpy(np.take_along_axis(psi, idx, axis=-1)).to(theta.device)

        rif_feats = torch.cat([rp, rq, theta, phi], axis=-1).view(B, M, num_samples, 4*2).contiguous() # B M S 4*2

    return cent_pts, cent_nms, xyz_feats, rif_feats, sm_pts, sm_feats