import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

import ipdb
from PIL import Image
import numpy as np
import torchvision
import os


# 1. basic feature extraction
class ImageFeatureExtractor(nn.Module):
  # 2D convs: 2 scale, layer 3 & 6
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


# 2. cost volume building
class CostVolume(nn.Module):
  def __init__(self):
        super(CostVolume, self).__init__()
  def warp_src_to_ref(self, src_feat_2d, depth_values, ref_in, ref_ex, src_in, src_ex):
      # this function is called for every img (ref & each src)

      # goal: src 2D pixel feature -> ref 3D feature
      # mapping from ref 3D -> src 3D -> src 2D
      
      # Note that the affine transformation is performed on COORDINATES rather than feaature VALUES of each pixel/point
      # therefore we need to construct variables to store the pixel coordinates, space coordinates respectively.
    b, c ,h, w = src_feat_2d.shape # shape same as ref
    # ipdb.set_trace()
    with torch.no_grad():
        # 1) ref pixel coords
        y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=src_feat_2d.device),
                                    torch.arange(0, w, dtype=torch.float32, device=src_feat_2d.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h * w), x.view(h * w) # [H*W]
        xy_homo = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W], homogeneous coord
        xy_homo = torch.unsqueeze(xy_homo, 0).repeat(b, 1, 1)  # [B, 3, H*W]

        # 2) ref pixel coord -> ref 3D coords: by inverse ref intrinsics & depth
        # ipdb.set_trace()
        
        xyz_ray = torch.matmul(torch.inverse(ref_in.detach().clone()), xy_homo) # without depth
        num_depth = depth_values.shape[1]
        ref_3D = xyz_ray.unsqueeze(1).repeat(1, num_depth, 1, 1) * depth_values.view(b, num_depth, 1,
                                                                                            1)  # [B, Ndepth, 3, H*W]

        # 3) ref 3D -> src 3D: R_src x R_ref(-1)
        relative_ex = torch.matmul(src_ex, torch.inverse(ref_ex.detach().clone())) # [B, 4, 4]
        relative_ex = relative_ex.unsqueeze(1).repeat(1, num_depth, 1, 1) # [B, Ndepth, 4, 4]
        rotation = relative_ex[:, :, :3, :3] # [B, Ndepth, 3, 3] 
        translation = relative_ex[:, :, :3, 3:] # [B, Ndepth, 3, 1] 
        src_3D = torch.matmul(rotation, ref_3D)+translation # [B, Ndepth, 3, H*W] 

        # 4) src 3D -> src 2D: by src intrinsics  
        src_in = src_in.unsqueeze(1).repeat(1, num_depth, 1, 1) # [B, Ndepth, 3, 3]
        src_2D_homo = torch.matmul(src_in, src_3D) # [B, Ndepth, 3, H*W] 
        src_2D = src_2D_homo[:, :, :2, :]/src_2D_homo[:, :, 2:, :] # [B, Ndepth, 2, H*W]

        # 5) feature warp from src 2D to ref 3D
        # with the coord matching of ref 3D to src 2D (src_2D stores the coord on src 2D, which is indexed by red 3D coord)
        grid_x_normalized = src_2D[:, :, 0, :] / ((w - 1) / 2) - 1
        grid_y_normalized = src_2D[:, :, 1, :] / ((h - 1) / 2) - 1  
        grid = torch.stack((grid_x_normalized, grid_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        
    warped_src_feat_volume = F.grid_sample(src_feat_2d, grid.view(b, num_depth * h, w, 2), mode='bilinear',
                                padding_mode='zeros')
    warped_src_feat_volume = warped_src_feat_volume.view(b, c, num_depth, h, w)
    # ipdb.set_trace()
    
    return warped_src_feat_volume


  def forward(self, depth_values, features_2d, cam_intrinsics, cam_extrinsics):

      ref_feature, src_features = features_2d[0], features_2d[1:]
      ref_in, src_ins = cam_intrinsics[0], cam_intrinsics[1:]
      ref_ex, src_exs = cam_extrinsics[0], cam_extrinsics[1:]
      num_depth = depth_values.shape[1]
      num_views = len(features_2d)

      # 1. differentiable homonograhy
      # 1) ref itself does not acquire any tranformation
      ref_feat_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
      volume_sum = ref_feat_volume 
      volume_sq_sum = ref_feat_volume ** 2
      del ref_feat_volume
      # 2) for every src view, warp src 2D -> ref 2D: affine transformation
      for src_feat_2d, src_in, src_ex in zip(src_features, src_ins, src_exs):
        warped_src_feat_volume = self.warp_src_to_ref(src_feat_2d, depth_values, ref_in, ref_ex, src_in, src_ex)
        volume_sum = volume_sum + warped_src_feat_volume
        volume_sq_sum =volume_sq_sum + warped_src_feat_volume.pow_(2)
        del warped_src_feat_volume

      # 2. cost volume calculation by variance
      # for each depth plane:
      # measure the disparity among all 3 views
      # at each pixel location, on each feature dimension.
      # (larger variance --> more difference --> low probability of belonging to the plane of that depth)
    #   cost_volume_by_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) # [B, C, Ndepth, H, W]
      cost_volume_by_variance = volume_sq_sum/num_views - (volume_sum/num_views)**2
  
      return cost_volume_by_variance


# 3. cost aggregation
class CostAggregation(nn.Module):
  def __init__(self):
        super(CostAggregation, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.out = nn.Conv3d(8, 1, 3, stride=1, padding=1)

  def channel_aggregation(self, x):
      conv0 = self.conv0(x)
      conv2 = self.conv2(self.conv1(conv0))
      conv4 = self.conv4(self.conv3(conv2))
      x = self.conv6(self.conv5(conv4))
      x = conv4 + self.conv7(x)
      x = conv2 + self.conv9(x)
      x = conv0 + self.conv11(x)
      x = self.out(x)
      return x

  def forward(self, cost_volume_by_variance):
    #2. squeeze channel dimension from 32 ->1 to do softmax on depth dimension
    cost_volume = self.channel_aggregation(cost_volume_by_variance) # [B, 1, Ndepth, H, W]
    cost_volume = cost_volume.squeeze(1) # [B, Ndepth, H, W]
    prob_volume = F.softmax(cost_volume, dim=1) # [B, Ndepth, H, W]

    return prob_volume

# 5. Depth Map Refinement
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, imgs, depth_inits, depth_values):
        toP = torchvision.transforms.ToPILImage()
        toT = torchvision.transforms.ToTensor()
        img_list = []
        depth_list = []
        for img, depth_init, depth_v in zip(imgs, depth_inits, depth_values): 
            img_save = toP(img)
            # img_save.save('/mnt/lustre/yslan/meng/MVSNet_pytorch/outputs/original.jpg')
            n, d_h, d_w = depth_init.shape
            resized_img = img_save.resize((d_w,d_h), Image.BILINEAR)
            # resized_img.save('/mnt/lustre/yslan/meng/MVSNet_pytorch/outputs/resized.jpg')
            # !!IMPORTANT: here is (w, h), not (h, w), BUT img[0] is (h, w), but object Image.size will show (w,h)
            # !!IMPORTANT_2: the resize() will not change the resized_img it self, so directly save resized_img will save the unresized one
        
            # convert back from Image obj to tensor, then concat with depth map
            # RBG -> [0,1] tensor, consistent with the passed in parameter 'img' (which is also 0-1)
            img_cat = toT(resized_img) # (w, h) 0~255 -> (3, h, w) 0~1
            img_list.append(img_cat.unsqueeze(0)) 

            # normalize depth map to 0~1
            # ipdb.set_trace()
            depth_normalized = (depth_init-depth_v.min())/(depth_v.max()-depth_v.min())
            depth_list.append(depth_normalized.unsqueeze(0))
        
        imgs_resized = torch.cat(img_list, 0).to(depth_inits.device)
        depth_inits_normalized = torch.cat(depth_list, 0).to(depth_inits.device)
        concat = torch.cat((imgs_resized, depth_inits_normalized), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        # ipdb.set_trace()
        # FIXME: depth_residual is not at the same scale as depth_normalized: it can have very large values
        depth_refined_norm = depth_inits_normalized + depth_residual

        depth_denorm_list = []
        for depth_norm, depth_v in zip(depth_refined_norm, depth_values):
            depth_denorm = depth_norm*(depth_v.max()-depth_v.min())+depth_v.min()
            depth_denorm_list.append(depth_denorm.unsqueeze(0))
        depth_refined = torch.cat(depth_denorm_list,0).to(depth_inits.device)
        
        return depth_refined


########################
class MVSNet2(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet2, self).__init__()
        self.feature_extractor = ImageFeatureExtractor() 
        self.cost_vol_builder = CostVolume()
        self.cost_aggregation = CostAggregation()
        self.refine = refine
        if self.refine:
            self.refine_network = RefineNet()
        
    def forward(self, imgs, camera_intrinsics, camera_extrinsics, depth_values):
        '''
        input:
        imgs: batch x nview x 3 x H x W  
        camera_intrinsics: batch x nview x 3 x 3
        camera_extrinsics: batch x nview x 4 x 4
        depth_values: batch x nview x (depth_max - depth_min)

        output:
        depth_map: H x W
        '''
        # # 0. save src imgs to check data correctness`
        # toP = torchvision.transforms.ToPILImage()
        # # os.mkdir('/mnt/lustre/yslan/meng/MVSNet_pytorch/outputs_checkdata', exist_ok=True)
        # for b, img_b in enumerate(imgs):
        #     for i, img in enumerate (img_b):
        #         img_save = toP(img)
        #         img_save.save('/mnt/lustre/yslan/meng/MVSNet_pytorch/outputs_feb19/bottles_ckpt14_pair_height/outputs_checkdata/{:0>3}_{}_{}.jpg'.format(batch_idx, b, i))

        imgs = torch.unbind(imgs, 1)
        cams_in = torch.unbind(camera_intrinsics, 1)
        cams_ex = torch.unbind(camera_extrinsics, 1)
        assert len(imgs) == len(cams_in)==3, "MUST have 3 imgs as in Problem3.8"
        
        # ipdb.set_trace()# check depth_values

        
        # 1. basic feature extraction
        features_2d = [self.feature_extractor(img) for img in imgs]
        
        # 2.cost volume building
        cost_volume_by_variance = self.cost_vol_builder(depth_values, features_2d, cams_in, cams_ex)
        
        # 3. cost volume aggregation & probability map
        prob_volume = self.cost_aggregation(cost_volume_by_variance)  # [B, Ndepth, H, W]
        # 3.1 photometric confidence
        with torch.no_grad():    
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = torch.arange(depth_values.shape[1], device=prob_volume.device, dtype=torch.float)
            depth_index = depth_index.unsqueeze(-1).unsqueeze(-1)
            depth_index = prob_volume*depth_index
            depth_index = depth_index.sum(1).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        
        # 4. depth map generation
        depth_values = depth_values.unsqueeze(-1).unsqueeze(-1) # [B, D, 1, 1]
        depth_estimate = prob_volume*depth_values
        depth_map = depth_estimate.sum(1) # sum across all depth

        # 5. Depth Map Refinement
        if self.refine:
            refined_depth = self.refine_network(imgs[0], depth_map.unsqueeze(1), depth_values)
            return {'depth': depth_map, "refined_depth": refined_depth, "photometric_confidence":photometric_confidence}

        return {'depth': depth_map, "photometric_confidence":photometric_confidence}   