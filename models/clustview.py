import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torchvision.models import resnet18, resnet34, resnet50

from pointnet2_ops import pointnet2_utils
from util.pointnet_util import index_points, square_distance, PointNetFeaturePropagation
from util.pcview import PCViews
from models.resent import ResNet


def Point2Patch(num_patches, patch_size, xyz):
    """
    Patch Partition in 3D Space
    Input:
        num_patches: number of patches, S
        patch_size: number of points per patch, k
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        knn_idx: [B, S, k]
    """

    # FPS the patch centroid out
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_patches).long()  # [B, S]
    centroid_xyz = index_points(xyz, fps_idx)  # [B, S, 3]

    # knn to group per patch
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k]

    return centroid_xyz, fps_idx, knn_idx


def cosine_distance(src, dst):
    """
    Calculate cosine distance between each two points.
    src: source points, [B, N, C]
    dst: target points, [B, M, C]
    Return: per-point cosine distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape

    # Normalize src and dst to unit vectors
    src_norm = src / (torch.norm(src, dim=-1, keepdim=True) + 1e-8)  # [B, N, C]
    dst_norm = dst / (torch.norm(dst, dim=-1, keepdim=True) + 1e-8)  # [B, M, C]

    # Compute cosine similarity: dot product of normalized vectors
    cos_sim = torch.matmul(src_norm, dst_norm.permute(0, 2, 1))  # [B, N, M]

    # Compute cosine distance
    cos_dist = 1 - cos_sim  # [B, N, M]

    return cos_dist


class BasicBlock(nn.Module):
    def __init__(self, in_channels, planes, bn_d=0.1, kernel_size=3, pooling=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, planes[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        B, S, C, H, W = x.size()

        out = self.conv1(x.view(-1, C, H, W))
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.pooling:
            out = self.pool2d(out)

        _, C, H, W = out.size()

        return out.view(B, S, C, H, W)


class View_Pooling(nn.Module):

    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels*4, out_channels, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())

    def forward(self, x):
        '''
        x's shape is (B, 6, 32, 64, 64)
        '''

        lr = torch.max(x[:, [0, 3, 6], :, :], 1)[0] # left and right
        fb = torch.max(x[:, [1, 4, 7], :, :], 1)[0] # front and back
        tb = torch.max(x[:, [2, 5, 8], :, :], 1)[0] # top and bottom
        al = torch.max(x, 1)[0]

        feat = torch.cat([al, lr, fb, tb], 1)
        feat = self.net(feat)

        return feat


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(mid_channels, out_channels)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchAbstraction(nn.Module):
    def __init__(self, num_patches, patch_size, in_channels, out_channels):
        super(PatchAbstraction, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed1 = nn.Sequential(nn.Conv1d(in_channels*2, out_channels, 1),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(inplace=True))

        self.embed2 = nn.Sequential(nn.Conv1d(out_channels*2, out_channels, 1),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, feature, xyz):
        """
        Input: xyz [B, S_, 3]
               features [B, S_, C]
        Return:
               centroid features [B, S, 3]
               avg features [B, S, C]
        """
        B, _, C = feature.shape
        centroid_xyz, centroid_idx, knn_idx = Point2Patch(self.num_patches, self.patch_size, xyz)

        centroid_feature = index_points(feature, centroid_idx)  # [B, S, C]
        grouped_feature = index_points(feature, knn_idx)  # [B, S, k, C]

        k = grouped_feature.shape[2]

        # Normalize
        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C)  # [B, S, k, C]
        groups = torch.cat((centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C), grouped_norm),
                           dim=-1)  # [B, S, k, 2C]

        groups = groups.reshape(-1, k, 2*C).permute(0, 2, 1) # [B*S, 2C, k]
        groups = self.embed1(groups) # [B*S, C, k]
        BS, C, k = groups.shape

        max_fea = torch.max(groups, 2, keepdim=True)[0]  # [B*S, C, 1]
        max_fea = torch.cat([max_fea.expand(-1, -1, k), groups], dim=1) # [B*S, 2C, k]
        max_fea = self.embed2(max_fea) # [B*S, C, k]
        max_fea = torch.max(max_fea, 2, keepdim=False)[0].reshape(B, -1, C)  # [B, S, C]
        # max_fea = torch.max(groups, 2, keepdim=True)[0].reshape(B, -1, C)  # [B*S, C, 1]

        return max_fea, centroid_xyz


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):

        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous())
        q, k, v = (qkv[0], qkv[1], qkv[2])

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ContextAttention(nn.Module):
    '''
    Content-based Transformer
    Args:
        dim (int): Number of input channels.
        local_size (int): The size of the local feature space.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''
    def __init__(self, dim, num_heads, local_size=16, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kmeans=False):
        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        Input: [B, S, C]
        Return: [B, S, C]
        '''

        B, S, C = x.shape
        nl = S // self.ls
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, h, S, c]

        q_pre = qkv[0].reshape(B * self.num_heads, S, C // self.num_heads).permute(0, 2, 1)  # [B*h, c, S]
        ntimes = int(math.log(nl, 2))
        q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B * self.num_heads, S)

        # balanced binary clustering
        for _ in range(ntimes):
            bh, d, n = q_pre.shape  # [B*h*2^n, c, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n // 2)  # [B*h*2^n, c, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1)  # [B*h*2^n, c, 2]

            q_avg = torch.nn.functional.normalize(q_avg.permute(0, 2, 1), dim=-1)
            q_norm = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1)

            q_scores = square_distance(q_norm, q_avg)  # [B*h*2^n, S/2^n, 2]
            # q_scores = cosine_distance(q_norm, q_avg)
            q_ratio = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)  # [B*h*2^n, S/2^n]
            q_idx = q_ratio.argsort()

            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)  # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size())  # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n // 2)  # [B*h*2^n, c, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')  # [B*h*2^(n+1), c, S/(2^(n+1))]

            # # Save indices for this iteration
            # idx_iter = q_idx_last.view(B, self.num_heads, -1)  # [B, h, S/2^iter]
            # idx_head_iter = idx_iter[0, 0].reshape(-1)
            # idx_head_iter = idx_head_iter.detach().cpu().numpy()
            # np.save(f'./render/{name}_idx{index}_stage_iter.npy', idx_head_iter)

        # clustering is performed independently in each head
        q_idx = q_idx_last.view(B, self.num_heads, S)  # [B, h, S]
        q_idx_rev = q_idx.argsort()  # [B, h, S]

        # cluster query, key, value
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())  # [3, B, h, S, c]
        qkv_pre = qkv.gather(dim=-2, index=q_idx)  # [3, B, h, S, d]
        q, k, v = rearrange(qkv_pre, 'qkv b h (nl ls) c -> qkv (b nl) h ls c', ls=self.ls)

        # MSA
        attn = (q - k) * self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = torch.einsum('bhld, bhld->bhld', attn, v)  # [B*(nl), h, ls, c]

        # merge and reverse
        out = rearrange(out, '(b nl) h ls c -> b h c (nl ls)', h=self.num_heads, b=B)  # [B, h, c, S]
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1, index=q_idx_rev).reshape(B, C, S).permute(0, 2, 1)  # [B, S, C]

        res = self.proj(res)  # [B, S, C]
        res = self.proj_drop(res)

        return res


class Block(nn.Module):
    def __init__(self, embed_dim, local_size, num_heads, drop_path=0.1, mlp_ratio=4.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, index_layrer=0):
        super(Block, self).__init__()

        self.norm1 = norm_layer(embed_dim)
        if index_layrer % 1 != 0:
            self.attn = ContextAttention(dim=embed_dim, local_size=local_size, num_heads=num_heads)
        else:
            self.attn = MultiheadAttention(dim=embed_dim, heads=num_heads, dropout=0.1)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(in_channels=embed_dim, mid_channels=int(embed_dim * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Basiclayer(nn.Module):
    def __init__(self, embed_dim, local_size, num_heads, drop_path, mlp_ratio, depth):
        super().__init__()

        self.blocks = nn.ModuleList([Block(embed_dim=embed_dim,
                                           local_size=local_size,
                                           num_heads=num_heads,
                                           drop_path=drop_path[i],
                                           mlp_ratio=mlp_ratio,
                                           index_layrer=i)
                                     for i in range(depth)])

    def forward(self, pc_feat, pos):

        for blk in self.blocks:
            pc_feat = blk(pc_feat)

        return pc_feat, pos


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim

        from models.dgcnn import DGCNN
        self.model = DGCNN()

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depth))]  # stochastic depth decay rule
        self.sample = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(len(cfg.depth)):
            sample = PatchAbstraction(num_patches=int(cfg.num_points / (cfg.down_ratio[i])),
                                      patch_size=cfg.patch_size,
                                      in_channels=cfg.embed_dim[i],
                                      out_channels=cfg.embed_dim[i+1])
            layer = Basiclayer(embed_dim=cfg.embed_dim[i+1],
                               local_size=cfg.local_size,
                               num_heads=cfg.num_heads,
                               drop_path=dpr[sum(cfg.depth[:i]):sum(cfg.depth[:i + 1])],
                               mlp_ratio=cfg.mlp_ratio,
                               depth=cfg.depth[i])

            self.layers.append(layer)
            self.sample.append(sample)

        self.pcview = PCViews()

        self.bin_num1 = [2, 4, 8, 16, 32]        # 32*32 -> 4*1024 8*512 16*256 32*128 / 1024*4 512*8 256*16 128*32 -> 4 8 16 32 128 256 512 1024
        self.bin_num2 = [1, 2, 4, 8, 16]         # 16*16
        self.bin_num3 = [1, 2, 4, 8]             # 8*8
        self.bin_num4 = [1, 2, 4]                # 4*4

        self.set_layer1 = BasicBlock(in_channels=1, planes=[cfg.proj_dim[0], cfg.proj_dim[0]], kernel_size=5, pooling=True)
        self.set_layer2 = BasicBlock(in_channels=cfg.proj_dim[0], planes=[cfg.proj_dim[1], cfg.proj_dim[1]], pooling=True)
        self.set_layer3 = BasicBlock(in_channels=cfg.proj_dim[1], planes=[cfg.proj_dim[2], cfg.proj_dim[2]], pooling=True)
        self.set_layer4 = BasicBlock(in_channels=cfg.proj_dim[2], planes=[cfg.proj_dim[3], cfg.proj_dim[3]], pooling=True)

        self.vp1 = View_Pooling(in_channels=cfg.proj_dim[0], out_channels=cfg.proj_dim[0])
        self.vp2 = View_Pooling(in_channels=cfg.proj_dim[1], out_channels=cfg.proj_dim[1])
        self.vp3 = View_Pooling(in_channels=cfg.proj_dim[2], out_channels=cfg.proj_dim[2])
        self.vp4 = View_Pooling(in_channels=cfg.proj_dim[3], out_channels=cfg.proj_dim[3])

        # self.img_model = ResNet(resnet18(pretrained=False), feat_dim=512)
        # self.img_model = ResNet(resnet34(pretrained=False), feat_dim=512)
        # self.img_model = ResNet(resnet50(pretrained=False), feat_dim=2048)

    def get_img(self, inpt):
        B = inpt.shape[0]
        imgs = self.pcview.get_img(inpt, 64)
        num_img = self.pcview.num_views

        _, H, W = imgs.shape
        imgs = imgs.reshape(B, num_img, -1)
        max = torch.max(imgs, -1, keepdim=True)[0]
        min = torch.min(imgs, -1, keepdim=True)[0]

        nor_img = (imgs - min) / (max - min + 0.0001)
        nor_img = nor_img.reshape(B, num_img, H, W)

        return nor_img

    def forward_img(self, inpt):
        # B 6 H W
        # img = self.get_img(inpt[:, :, :3])
        # img_feat = self.img_model(img)

        img = self.get_img(inpt[:, :, :3]).unsqueeze(2)
        img = self.set_layer1(img)
        img_x1 = self.vp1(img)

        img = self.set_layer2(img)
        img_x2 = self.vp2(img)

        img = self.set_layer3(img)
        img_x3 = self.vp3(img)

        img = self.set_layer4(img)
        img_x4 = self.vp4(img)

        view_proj1 = []
        view_proj2 = []
        view_proj3 = []
        view_proj4 = []
        B, C, H, W = img_x1.size()
        for num_bin in self.bin_num1:
            z = img_x1.view(B, C, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            view_proj1.append(z)
            z = img_x1.view(B, C, -1, num_bin)
            z = z.mean(3) + z.max(3)[0]
            view_proj1.append(z)

        B, C, H, W = img_x2.size()
        for num_bin in self.bin_num2:
            z = img_x2.view(B, C, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            view_proj2.append(z)
            z = img_x2.view(B, C, -1, num_bin)
            z = z.mean(3) + z.max(3)[0]
            view_proj2.append(z)

        B, C, H, W = img_x3.size()
        for num_bin in self.bin_num3:
            z = img_x3.view(B, C, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            view_proj3.append(z)
            z = img_x3.view(B, C, -1, num_bin)
            z = z.mean(3) + z.max(3)[0]
            view_proj3.append(z)

        B, C, H, W = img_x4.size()
        for num_bin in self.bin_num4:
            z = img_x4.view(B, C, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            view_proj4.append(z)
            z = img_x4.view(B, C, -1, num_bin)
            z = z.mean(3) + z.max(3)[0]
            view_proj4.append(z)

        view_proj1 = torch.cat(view_proj1, dim=2)  # b c bin
        view_proj2 = torch.cat(view_proj2, dim=2)  # b c bin
        view_proj3 = torch.cat(view_proj3, dim=2)  # b c bin
        view_proj4 = torch.cat(view_proj4, dim=2)  # b c bin

        img_feat = torch.cat([view_proj1.max(2)[0],
                              view_proj2.max(2)[0],
                              view_proj3.max(2)[0],
                              view_proj4.max(2)[0], ], dim=-1)

        return img_feat

    def forward_pc(self, inpt):

        pc_feat, xyz = inpt, inpt
        for i, (layer) in enumerate(self.layers):
            pc_feat, xyz = self.sample[i](pc_feat, xyz)
            pc_feat, xyz = layer(pc_feat, xyz)

            # pos_np = xyz.detach().cpu().numpy()
            # np.save('../Assistance/PointRenderer/data/' + name + '_pos_stage%d' %i, pos_np[0])

        return pc_feat

    def forward(self, x):

        img_feat = self.forward_img(x)
        pc_feat = self.forward_pc(x)

        return pc_feat, img_feat


class ClustView_cls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = TransformerEncoder(cfg)

        point_dim = cfg.embed_dim[-1]
        sum_proj_dim = sum(cfg.proj_dim)

        self.mlp_head = nn.Sequential(nn.Linear(point_dim*2+sum_proj_dim, 256, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Dropout(p=cfg.dropout))

        self.cls_head = nn.Linear(256, cfg.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x -> B N C
        pc_feat, img_feat = self.encoder(x)  # B N C
        pc_feat = torch.cat([torch.max(pc_feat, dim=1)[0], torch.mean(pc_feat, dim=1)], dim=-1)
        
        fea = torch.cat([pc_feat, img_feat], dim=-1)  # B C
        fea = self.mlp_head(fea)
        fea = self.cls_head(fea)

        return fea


class ClustView_partseg(nn.Module):
    def __init__(self, cfg, seg_num_all):
        super().__init__()
        self.seg_num_all = seg_num_all
        self.encoder = TransformerEncoder(cfg)

        self.fp1 = PointNetFeaturePropagation(in_channels=cfg.embed_dim[-1]+cfg.embed_dim[-2], mlp=[512])               # 1024 -> 512 + 512
        self.fp2 = PointNetFeaturePropagation(in_channels=512+cfg.embed_dim[-3], mlp=[384])                             # 512 + 256
        self.fp3 = PointNetFeaturePropagation(in_channels=384+cfg.embed_dim[-4], mlp=[256])                             # 384 + 128
        self.fp4 = PointNetFeaturePropagation(in_channels=256+cfg.embed_dim[-5], mlp=[128])                             # 256 + 64
        self.fp5 = PointNetFeaturePropagation(in_channels=128, mlp=[128])                                               # 256 + 64

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.1))

        self.dpr = nn.Conv1d(128*3+64, 128, 1)
        self.seg = nn.Conv1d(128, seg_num_all, 1)
        self.bn1  = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(cfg.dropout)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        B, C, N = x.shape
        x = x.permute(0, 2, 1) # B N C

        list_feat, list_proj = self.encoder(x)

        point1_fea, point1_xyz = list_feat[0][0], list_feat[0][1] # B N C
        point2_fea, point2_xyz = list_feat[1][0], list_feat[1][1]
        point3_fea, point3_xyz = list_feat[2][0], list_feat[2][1]
        point4_fea, point4_xyz = list_feat[3][0], list_feat[3][1]
        point5_fea, point5_xyz = list_feat[4][0], list_feat[4][1]

        point4_fea = self.fp1(point4_xyz, point5_xyz, point4_fea, point5_fea)
        point3_fea = self.fp2(point3_xyz, point4_xyz, point3_fea, point4_fea)
        point2_fea = self.fp3(point2_xyz, point3_xyz, point2_fea, point3_fea)
        point1_fea = self.fp4(point1_xyz, point2_xyz, point1_fea, point2_fea)
        point0_fea = self.fp5(x, point1_xyz, None, point1_fea)

        x_max = torch.max(point0_fea, 1)[0] # B N C -> B C
        x_avg = torch.mean(point0_fea, 1)
        x_max = x_max.view(B, -1).unsqueeze(-1)
        x_avg = x_avg.view(B, -1).unsqueeze(-1)
        x_cls = cls_label.view(B, 16, -1)
        x_cls = self.label_conv(x_cls)

        x = torch.cat([x_max, x_avg, x_cls], dim=1).repeat(1, 1, N)
        x = torch.cat([x, point0_fea.permute(0, 2, 1)], dim=1)
        x = self.drop(self.relu(self.bn1(self.dpr(x)))) # x: [B, C, N]
        x = self.seg(x)

        return x


class ClustView_semseg(nn.Module):
    def __init__(self, cfg, seg_num_all):
        super().__init__()
        self.seg_num_all = seg_num_all
        self.encoder = TransformerEncoder(cfg)

        self.fp1 = PointNetFeaturePropagation(in_channels=192+96, mlp=[288, 288])           # 1024 -> 512 + 512
        self.fp2 = PointNetFeaturePropagation(in_channels=288+48, mlp=[336, 336])           # 512 + 256
        self.fp3 = PointNetFeaturePropagation(in_channels=336+24, mlp=[360, 360])           # 384 + 128
        self.fp4 = PointNetFeaturePropagation(in_channels=360, mlp=[360, 128])              # 256 + 64

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.1))

        self.dpr = nn.Conv1d(128*3+64+
                             cfg.proj_dim[0]+
                             cfg.proj_dim[1]+
                             cfg.proj_dim[2]+
                             cfg.proj_dim[3], 128, 1)
        self.seg = nn.Conv1d(128, seg_num_all, 1)
        self.bn1  = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(cfg.dropout)
        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        B, C, N = x.shape
        x = x.permute(0, 2, 1) # B N C

        list_feat, list_proj = self.encoder(x)

        point1_fea, point1_xyz = list_feat[0][0], list_feat[0][1]   # B N C
        point2_fea, point2_xyz = list_feat[1][0], list_feat[1][1]
        point3_fea, point3_xyz = list_feat[2][0], list_feat[2][1]
        point4_fea, point4_xyz = list_feat[3][0], list_feat[3][1]

        point3_fea = self.fp1(point3_xyz, point4_xyz, point3_fea, point4_fea)
        point2_fea = self.fp2(point2_xyz, point3_xyz, point2_fea, point3_fea)
        point1_fea = self.fp3(point1_xyz, point2_xyz, point1_fea, point2_fea)
        point0_fea = self.fp4(x, point1_xyz, None, point1_fea)

        x_max = torch.max(point0_fea, 1)[0] # B N C -> B C
        x_avg = torch.mean(point0_fea, 1)
        x_max = x_max.view(B, -1).unsqueeze(-1)
        x_avg = x_avg.view(B, -1).unsqueeze(-1)
        x_img = list_proj.view(B, -1).unsqueeze(-1)
        x_cls = cls_label.view(B, 16, -1)
        x_cls = self.label_conv(x_cls)

        x = torch.cat([x_max, x_avg, x_cls, x_img], dim=1).repeat(1, 1, N)
        x = torch.cat([x, point0_fea.permute(0, 2, 1)], dim=1)
        x = self.drop(self.relu(self.bn1(self.dpr(x))))  # x: [B, C, N]
        x = self.seg(x)

        return x
