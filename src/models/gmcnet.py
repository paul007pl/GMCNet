import math
from textwrap import indent
import numpy
import sys
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# from pytorch3d.ops.knn import knn_points


from visu_utils import visualize
from train_utils import *
from model_utils import clones, attention, knn_point, knn, get_edge_features, angle, estimate_fpfh, sinkhorn, Conv1DBNReLU, square_distance, get_cluster_feats, sample_and_group_feats


sys.path.append("../utils")
from mm3d_pn2 import three_nn, three_interpolate, furthest_point_sample, gather_points, grouping_operation
# from metrics import cd

_EPS = 1e-6  # To prevent division by zero


def get_us_feats(feats, src_pts, tgt_pts):
    idx, weight = three_nn_upsampling(tgt_pts, src_pts)
    feats = three_interpolate(feats, idx, weight)
    return feats


def three_nn_upsampling(target_points, source_points):
    dist, idx = three_nn(target_points, source_points)
    dist = torch.max(dist, torch.ones(1).cuda()*1e-10)
    norm = torch.sum((1.0/dist), 2, keepdim=True)
    norm = norm.repeat(1,1,3)
    weight = (1.0/dist) / norm
    return idx, weight


class Normal_Estimation(nn.Module):
    def __init__(self, radius=0.1, max_nn=30):
        super(Normal_Estimation, self).__init__()
        self.radius_normal = radius
        self.nn_normal = max_nn

    def forward(self, xyz):
        xyz = xyz.cpu().numpy()
        res = np.zeros((xyz.shape[0], xyz.shape[1], 3))

        for i in range(len(xyz)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[i])
            # estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=self.nn_normal))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            # pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=100))
            # res[i] = pcd_fpfh.data
            res[i] = np.array(pcd.normals)

        res = torch.from_numpy(res).float().cuda()
        return res


class Point_Transformer(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, share_planes=8, pt_planes=8, k=10):
        super(Point_Transformer, self).__init__()
        self.share_planes = share_planes
        self.k = k
        
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv1 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)

        self.conv_w = nn.Sequential(
                            nn.Conv2d(mid_planes*k, mid_planes//share_planes, kernel_size=1, bias=False),
                            nn.ReLU(inplace=False), 
                            nn.Conv2d(mid_planes//share_planes, k*mid_planes//share_planes, kernel_size=1))
        # self.activation_fn = nn.ReLU(inplace=False)
        
        self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

        self.pt_conv = nn.Sequential(
                            nn.Conv2d(pt_planes, mid_planes, kernel_size=1, bias=False), 
                            nn.ReLU(inplace=False), 
                            nn.Conv2d(mid_planes, mid_planes, kernel_size=1, bias=False))

    def forward(self, input):
        # # B C M S; B M 3; B 3 M S
        # sm_feats, cent_pts, sm_pts = input

        # B C1 M S; B C2 M S;
        sm_feats, sm_ppfs = input
        batch_size, _, num_points, num_samples = sm_feats.size()
        
        xn = F.relu(self.conv0(sm_feats)) # B C M S

        x = xn[:, :, :, 0, None] # B C M 1
        identity = x
        x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

        # pts = cent_pts.transpose(1,2).contiguous().unsqueeze(-1) # B 3 M 1
        # ptsn = torch.sum(torch.square(pts - sm_pts), dim=1, keepdim=True) # B 1 M S
        # ptf = self.pt_conv(ptsn) # B C M S

        ptf = self.pt_conv(sm_ppfs)

        x3 += ptf
        xfs = x1 - x2 + ptf # B C M S

        xfs = xfs.transpose(2, 3).contiguous().view(batch_size, -1, 1, num_points) # B CS 1 M
        w = F.softmax(self.conv_w(xfs).view(batch_size, -1, num_samples, num_points), dim=2).transpose(2, 3).contiguous() # B C M S
        w = w.repeat(1, self.share_planes, 1, 1) # B C M S

        out = w * x3 # B C M S
        out = F.relu(torch.sum(out, dim=3, keepdim=True)) # B C M 1
        out = (self.conv_out(out) + identity).squeeze(3) # B C M
        
        return out


class Robust_RPM_SVDHead(nn.Module):
    def __init__(self, alpha=0.20, beta=1.0, num_niters=5, use_weighted_procrustes=False, inlier_ratio=0.33, num_seeds=10, sub_set_ratio=0.2):
        super(Robust_RPM_SVDHead, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_iters = num_niters
        
        self.use_weighted_procrustes = use_weighted_procrustes
        self.inlier_ratio = inlier_ratio
        self.num_seeds = num_seeds
        self.sub_set_ratio = sub_set_ratio

        # bin_score = nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

    def regular_scores(self, scores):
        scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
        scores = torch.where(torch.isinf(scores), torch.ones_like(scores), scores)
        return scores

    def forward(self, src_embedding, tgt_embedding, src, tgt, learned_alpha=None, learned_beta=None, prefix="train"):
        # B C N; ; B 3 N;
        batch_size, num_dims, num_points = src_embedding.size()

        num_dims = src_embedding.size(1) # 512
        # scores = torch.matmul(F.normalize(src_embedding.transpose(2, 1).contiguous(), dim=2), F.normalize(tgt_embedding, dim=1))
        # scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(num_dims)

        feat_distance = square_distance(src_embedding.transpose(2, 1).contiguous(), 
                                   tgt_embedding.transpose(1, 2).contiguous()) / math.sqrt(num_dims)
        if (learned_alpha is None) or (learned_beta is None):
            scores = self.beta * (self.alpha - feat_distance)
        else:
            if isinstance(learned_alpha, float):
                scores = -learned_beta[:, None, None] * (feat_distance - learned_alpha)
            else:
                scores = -learned_beta[:, None, None] * (feat_distance - learned_alpha[:, None, None])
        scores = sinkhorn(scores, num_iters=self.num_iters)
        # scores = self.regular_scores(scores)

        if not self.use_weighted_procrustes:
            scores = torch.exp(scores[:, :-1, :-1])

            # src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
            src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous()) / (torch.sum(scores, dim=2).unsqueeze(1) + _EPS)

            # src_centered = src - src.mean(dim=2, keepdim=True)
            # src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

            conf = torch.sum(scores, 2).unsqueeze(1) / (torch.sum(torch.sum(scores, 2, keepdim=True), 1, keepdim=True) + _EPS) # B 1 N

            cent_src = (src * conf).sum(dim=2, keepdim=True) # B 3 1
            cent_src_corr = (src_corr * conf).sum(dim=2, keepdim=True)
            # cent_src = (src).sum(dim=2, keepdim=True) # B 3 1
            # cent_src_corr = (src_corr).sum(dim=2, keepdim=True)

            src_centered = src - cent_src # B 3 N
            src_corr_centered = src_corr - cent_src_corr

            # H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
            H = torch.matmul(src_centered * conf, src_corr_centered.transpose(2, 1).contiguous())
            # b x 3 x 3

            U, S, V = torch.svd(H.cpu(), some=False, compute_uv=True) # Bx3x3, _, Bx3x3
            U = U.to(H.device)
            V = V.to(H.device)
            
            R_pos = V @ U.transpose(1, 2)

            V_neg = V.clone()
            V_neg[:, :, 2] *= -1
            R_neg = V_neg  @ U.transpose(1, 2)
            R = torch.where(torch.det(R_pos)[:, None, None] > 0, R_pos, R_neg)
            assert torch.all(torch.det(R) > 0)

            # t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True) # B 3 1
            t = torch.matmul(-R, cent_src) + cent_src_corr # B 3 1

            bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device) # B 1 4
            T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
        
        else:
            num_inliers = int(num_points * self.inlier_ratio)
            num_samples = int(num_inliers * self.sub_set_ratio)

            scores_no_padded = torch.exp(scores[:, :-1, :-1]) # B N+1 M+1
            src_inlier_weights, src_corr_idx = torch.max(scores_no_padded, dim=-1) # B N
            # _, src_corr_pad_idx = torch.max(scores[:, :-1, :], dim=-1) # B N

            tgt_inlier_weights, tgt_corr_idx = torch.max(scores_no_padded, dim=1) # B M

            scores = scores_no_padded

            src_corr = torch.gather(tgt, dim=-1, index=src_corr_idx.unsqueeze(1).repeat(1,3,1)) # B 3 N
            tgt_corr = torch.gather(src, dim=-1, index=tgt_corr_idx.unsqueeze(1).repeat(1,3,1)) # B 3 M

            src_inlier_weights, src_inlier_idx = torch.topk(src_inlier_weights, k=num_inliers, dim=1, sorted=False) # B I
            tgt_inlier_weights, tgt_inlier_idx = torch.topk(tgt_inlier_weights, k=num_inliers, dim=1, sorted=False)

            src = torch.gather(src, dim=-1, index=src_inlier_idx.unsqueeze(1).repeat(1,3,1))
            src_corr = torch.gather(src_corr, dim=-1, index=src_inlier_idx.unsqueeze(1).repeat(1,3,1))
            tgt = torch.gather(tgt, dim=-1, index=tgt_inlier_idx.unsqueeze(1).repeat(1,3,1))
            tgt_corr = torch.gather(tgt_corr, dim=-1, index=tgt_inlier_idx.unsqueeze(1).repeat(1,3,1))
            
            inlier_weights = torch.cat([src_inlier_weights, tgt_inlier_weights], dim=-1) # B (N+M)
            inlier_weights = inlier_weights.unsqueeze(1) / (torch.sum(inlier_weights.unsqueeze(1), dim=-1, keepdim=True) + _EPS) # B 1 N

            src_inliers = torch.cat([src, tgt_corr], dim=-1)
            src_inliers_corr = torch.cat([src_corr, tgt], dim=-1)

            conf_inliers = inlier_weights

            cent_src = (src_inliers * conf_inliers).sum(dim=2, keepdim=True) # B 3 1
            cent_src_corr = (src_inliers_corr * conf_inliers).sum(dim=2, keepdim=True)
            src_centered = src_inliers - cent_src # B 3 N
            src_corr_centered = src_inliers_corr - cent_src_corr

            H = torch.matmul(src_centered * conf_inliers, src_corr_centered.transpose(2, 1).contiguous())
            U, S, V = torch.svd(H.cpu(), some=False, compute_uv=True) # Bx3x3, _, Bx3x3
            U = U.to(H.device)
            V = V.to(H.device)

            R_pos = V @ U.transpose(1, 2)

            V_neg = V.clone()
            V_neg[:, :, 2] *= -1
            R_neg = V_neg  @ U.transpose(1, 2)
            R = torch.where(torch.det(R_pos)[:, None, None] > 0, R_pos, R_neg)
            assert torch.all(torch.det(R) > 0)

            t = torch.matmul(-R, cent_src) + cent_src_corr # B 3 1

            bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device) # B 1 4
            T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)

        return T, scores


class FeatMultiFusion(nn.Module):
    def __init__(self, knn=[5, 5, 5], down_sample_list=[2, 4], feature_size_list=[128, 128, 32, 8], descriptor_size=128, rif="ppf", rif_only=False):
        super().__init__()

        self.rif = rif
        self.rif_only = rif_only

        if rif == "ppf":
            ri_feat_size = 4
            first_ri_feat_size = ri_feat_size*knn[0]

        elif rif == "fpfh":
            ri_feat_size = 4
            first_ri_feat_size = 33
        elif rif == "rri":
            ri_feat_size = 8
            first_ri_feat_size = ri_feat_size*knn[0]
        else:
            raise Exception('Either ppf, fpfh or rri')
        
        self.CBR0_RI = Conv1DBNReLU(first_ri_feat_size, 64)
        self.CBR_Res_RI = Conv1DBNReLU(64, feature_size_list[0])

        if self.rif_only:
            self.CBR1_RI = Conv1DBNReLU(first_ri_feat_size, 64)
            pt_feat_size = ri_feat_size

        else:
            self.CBR0_RE = Conv1DBNReLU(6*knn[0], 64)
            pt_feat_size = ri_feat_size + 6

        self.CBR01_F = Conv1DBNReLU(128, 128)
        self.CBR02_F = Conv1DBNReLU(128, feature_size_list[1])
        self.PT0 = Point_Transformer(feature_size_list[1], feature_size_list[1]//2, feature_size_list[1], 
                                        k=knn[0], pt_planes=pt_feat_size)

        self.PT1 = Point_Transformer(feature_size_list[0]+feature_size_list[1], (feature_size_list[0]+feature_size_list[1])//2, 128, 
                                        k=knn[1], pt_planes=pt_feat_size)
        self.CBR11_F = Conv1DBNReLU(128, 64)
        self.CBR12_F = Conv1DBNReLU(64, feature_size_list[2])

        self.PT2 = Point_Transformer(feature_size_list[2], feature_size_list[2]//2, 32, k=knn[2], pt_planes=pt_feat_size)
        self.CBR21_F = Conv1DBNReLU(32, 16)
        self.CBR22_F = Conv1DBNReLU(16, feature_size_list[3])

        self.fuse_proj = nn.Conv1d(feature_size_list[0]+feature_size_list[1]+feature_size_list[2]+feature_size_list[3], 
                                        descriptor_size, 1, bias=False)

        self.knn = knn
        self.down_sample_list = down_sample_list

    def forward(self, pts, nms=None):
        batch_size, num_points, _ = pts.size()
        
        _, _, delta_pts_feats, rif_feats, sm_pts, _ = sample_and_group_feats(-1, self.knn[0], pts, nms, rif=self.rif)

        if not self.rif_only:
            sm_cluster_feats0 = torch.cat([sm_pts, delta_pts_feats, rif_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 10 N k
        else:
            sm_cluster_feats0 = rif_feats.permute(0, 3, 1, 2).contiguous() # B 4 N k
        # sm_cluster_feats0 = torch.cat([sm_pts, delta_pts_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 6 N k
        if self.rif == "fpfh":
            rif_feats = estimate_fpfh(pts, nms) # B 33 N
        else:
            rif_feats = rif_feats.view(batch_size, num_points, -1).transpose(-1, -2).contiguous() # B 4k N
        delta_pts_feats = delta_pts_feats.view(batch_size, num_points, -1).transpose(-1, -2).contiguous() # B 3k N
        pts_feats = sm_pts.view(batch_size, num_points, -1).transpose(-1, -2).contiguous() # B 3k N

        ri_feats0 = self.CBR0_RI(rif_feats) # B 64 N
        unary_feats = self.CBR_Res_RI(ri_feats0) # B 128 N
        
        if not self.rif_only:
            re_feats = torch.cat([pts_feats, delta_pts_feats], dim=1) # B 6k N
            re_feats = self.CBR0_RE(re_feats) # B 64 N

            fuse_feats = torch.cat([ri_feats0, re_feats], dim=1) # B 128 N
        else:
            ri_feats1 = self.CBR1_RI(rif_feats) # B 64 N

            fuse_feats = torch.cat([ri_feats0, ri_feats1], dim=1) # B 128 N

        fuse_feats = self.CBR02_F(self.CBR01_F(fuse_feats)) # B C N
        sm_fuse_feats0 = get_cluster_feats(fuse_feats, pts, self.knn[0]) # B C N k
        fuse_feats0 = self.PT0([sm_fuse_feats0, sm_cluster_feats0]) # B C N
        
        
        ##########
        cent_pts1, cent_nms1, delta_pts_feats, rif_feats, sm_pts, sm_fuse_feats1 = sample_and_group_feats(num_points//self.down_sample_list[0], 
                                                                self.knn[1], pts, nms, torch.cat([unary_feats, fuse_feats0], dim=1), rif=self.rif)
        if not self.rif_only:
            sm_cluster_feats1 = torch.cat([sm_pts, delta_pts_feats, rif_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 10 M k
        else:
            sm_cluster_feats1 = rif_feats.permute(0, 3, 1, 2).contiguous() # B 4 M k
        # sm_cluster_feats1 = torch.cat([sm_pts, delta_pts_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 6 M k
        fuse_feats1 = self.CBR12_F(self.CBR11_F(self.PT1([sm_fuse_feats1, sm_cluster_feats1])))

        ##########
        cent_pts2, cent_nms2, delta_pts_feats, rif_feats, sm_pts, sm_fuse_feats2 = sample_and_group_feats(num_points//self.down_sample_list[1], 
                                                                                    self.knn[2], cent_pts1, cent_nms1, fuse_feats1, rif=self.rif)
        if not self.rif_only:
            sm_cluster_feats2 = torch.cat([sm_pts, delta_pts_feats, rif_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 10 M k
        else:
            sm_cluster_feats2 = rif_feats.permute(0, 3, 1, 2).contiguous() # B 4 M k
        # sm_cluster_feats2 = torch.cat([sm_pts, delta_pts_feats], dim=-1).permute(0, 3, 1, 2).contiguous() # B 6 M k
        fuse_feats2 = self.CBR22_F(self.CBR21_F(self.PT2([sm_fuse_feats2, sm_cluster_feats2])))

        fuse_feats1 = get_us_feats(fuse_feats1, cent_pts1, pts)
        fuse_feats2 = get_us_feats(fuse_feats2, cent_pts2, pts)

        feats = self.fuse_proj(torch.cat([unary_feats, fuse_feats0, fuse_feats1, fuse_feats2], dim=1))

        return feats


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        super().__init__()

        # self._logger = logging.getLogger(self.__class__.__name__)

        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.GroupNorm(16, 1024), nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512), nn.GroupNorm(16, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.GroupNorm(16, 256), nn.ReLU(),
            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        # self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        query = query.transpose(1,2).contiguous()
        key = key.transpose(1,2).contiguous()
        value = value.transpose(1,2).contiguous()

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        return x.transpose(1,2).contiguous()


class Cross_Attention(nn.Module):
    def __init__(self, descriptor_size, heads=4):
        super().__init__()
        self.attn = MultiHeadedAttention(heads, descriptor_size)
        self.ff = nn.Sequential(
                            nn.Conv1d(2*descriptor_size, 2*descriptor_size, 1),
                            nn.Conv1d(2*descriptor_size, descriptor_size, 1)
                            )
    def forward(self, desc1, desc2):
        desc1_ca = desc1 + self.ff(torch.cat([desc1, self.attn(desc1, desc2, desc2)], dim=1))
        desc2_ca = desc2 + self.ff(torch.cat([desc2, self.attn(desc2, desc1, desc1)], dim=1))
        return desc1_ca, desc2_ca


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.knn = [int(i) for i in args.knn_list.split(',')]
        self.down_sample_list = [int(i) for i in args.down_sample_list.split(',')]
        self.feature_size_list = [int(i) for i in args.feature_size_list.split(',')]

        self.use_cycle = args.use_cycle_loss
        self.weight_mse = args.use_mse_loss
        self.weight_cd = args.use_cd_loss
        self.weight_inliers = args.use_inliers_loss
        self.reduction = args.reductioin
        self.use_annealing = args.use_annealing
        self.rif_only = args.rif_only
        self.rif = args.rif_feature

        self.predict_inliers = args.predict_inliers

        self.emb_nm = Normal_Estimation()

        if self.use_annealing:
            self.weights_net = ParameterPredictionNet(weights_dim=[0])

        self.EncoderD = FeatMultiFusion(knn=self.knn, down_sample_list=self.down_sample_list, feature_size_list=self.feature_size_list, 
                                        descriptor_size=args.descriptor_size, rif=self.rif, rif_only=self.rif_only)
        
        if self.predict_inliers:
            self.CrossAttention = Cross_Attention(descriptor_size=args.descriptor_size)
            self.InlierPredict = nn.Sequential(
                                            nn.Conv1d(args.descriptor_size, args.descriptor_size//2, 1),
                                            nn.Conv1d(args.descriptor_size//2, 1, 1))

        self.RobustPointer = Robust_RPM_SVDHead(use_weighted_procrustes=args.use_weighted_procrustes)

    def forward(self, pts1, pts2, T_gt, prefix="train"):
        batch_size, num_points1, _ = pts1.size()
        _, num_points2, _ = pts2.size()
        
        self.pts1 = pts1
        self.pts2 = pts2

        if self.use_annealing:
            beta, alpha = self.weights_net([pts1, pts2])
        else:
            beta, alpha = None, None

        if self.rif == "ppf" or self.rif == "fpfh":
            nms1 = self.emb_nm(pts1)
            nms2 = self.emb_nm(pts2)
        elif self.rif == "rri":
            nms1 = None
            nms2 = None
        else:
            raise Exception('Either ppf, fpfh or rri')

        mean_pts1 = pts1.mean(dim=1, keepdim=True) # B 1 3
        mean_pts2 = pts2.mean(dim=1, keepdim=True)

        pt_feats1 = pts1 - mean_pts1 # B N 3
        pt_feats2 = pts2 - mean_pts2

        feats1 = self.EncoderD(pt_feats1, nms1) # B 128 N
        feats2 = self.EncoderD(pt_feats2, nms2)

        src, tgt = self.pts1.transpose(1,2), self.pts2.transpose(1,2)
        loss = 0

        self.T_12, self.scores12 = self.RobustPointer(feats1, feats2, src, tgt, alpha, beta)
        if prefix=="test":
            return self.T_12
        else:
            self.T_gt = T_gt
            

            if self.weight_mse > 0:
                loss += (rotation_geodesic_error(self.T_12[:, :3, :3], T_gt[:, :3, :3]) 
                            + translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])).mean() * self.weight_mse

            if self.weight_inliers > 0:
                loss += ((1.0 - torch.sum(self.scores12, dim=2)).mean() + (1.0 - torch.sum(self.scores12, dim=1)).mean()) * self.weight_inliers

            if self.weight_cd > 0:
                loss += (F.l1_loss(self.pts1 @ self.T_12[:, :3, :3].transpose(1, 2) + self.T_12[:, :3, 3].unsqueeze(1),
                                     self.pts1 @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1))) * self.weight_cd

            if self.use_cycle:
                self.T_21, self.scores21 = self.RobustPointer(feats2, feats1, tgt, src, alpha, beta)
                T_gt_inv = torch.inverse(T_gt)
                
                if self.weight_mse > 0:
                    loss += (rotation_geodesic_error(self.T_21[:, :3, :3], T_gt_inv[:, :3, :3]) 
                            + translation_error(self.T_21[:, :3, 3], T_gt_inv[:, :3, 3])).mean() * self.weight_mse

                if self.weight_inliers > 0:
                    loss += ((1.0 - torch.sum(self.scores21, dim=2)).mean() + (1.0 - torch.sum(self.scores21, dim=1)).mean()) * self.weight_inliers
                
                if self.weight_cd > 0:
                    loss += (F.l1_loss(self.pts2 @ self.T_21[:, :3, :3].transpose(1, 2) + self.T_21[:, :3, 3].unsqueeze(1),
                                     self.pts2 @ T_gt_inv[:, :3, :3].transpose(1, 2) + T_gt_inv[:, :3, 3].unsqueeze(1))) * self.weight_cd

            self.r_err = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
            self.t_err = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
            # self.rmse = rmse_loss(self.pts1[:, :100], self.T_12, T_gt)
            self.rmse = rmse_loss(self.pts1, self.T_12, T_gt)

            self.mse = (rotation_geodesic_error(self.T_12[:, :3, :3], T_gt[:, :3, :3]) + translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3]))
            # self.coarse_trans_loss = translation_error(pred_relative_trans.transpose(1,2), gt_relative_trans.transpose(1,2))

        return loss, self.r_err, self.t_err, self.rmse, self.mse

    def visualize(self, i):
        init_r_err = torch.acos((self.T_gt[i, :3, :3].trace() - 1) / 2) * 180 / math.pi
        init_t_err = torch.norm(self.T_gt[i, :3, 3])
        eye = torch.eye(4).unsqueeze(0).to(self.T_gt.device)
        init_rmse = rmse_loss(self.pts1[i:i+1], eye, self.T_gt[i:i+1])[0]
        pts1_trans = self.pts1[i] @ self.T_12[i, :3, :3].T + self.T_12[i, :3, 3]
        fig = visualize([self.pts1[i], self.gamma1[i], self.pi1[i], self.mu1[i], self.sigma1[i],
                            self.pts2[i], self.gamma2[i], self.pi2[i], self.mu2[i], self.sigma2[i],
                            pts1_trans, init_r_err, init_t_err, init_rmse,
                            self.r_err[i], self.t_err[i], self.rmse[i]])
        return fig


    def get_transform(self):
        return self.T_12, self.scores12

