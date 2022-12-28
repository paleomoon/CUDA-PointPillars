import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(False)

class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        inputs = inputs.transpose(1,3)
        x = self.linear(inputs)
        x = self.norm(x)
        x = F.relu(x)
        x = x.transpose(1,3).squeeze(0)
        x_max = torch.max(x, dim=1)[0]
        return x_max

class PillarVFE_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_x = 0.16
        self.voxel_y = 0.16
        self.voxel_z = 4
        self.x_offset = self.voxel_x / 2
        self.y_offset = self.voxel_y / 2 - 39.68
        self.z_offset = self.voxel_z / 2 - 3

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.floor(actual_num)
        paddings_indicator =  actual_num - max_num
        paddings_indicator  =  paddings_indicator  > 0
        return paddings_indicator

    def forward(self, voxel_features, voxel_num_points, coords, voxel_count_data):
        voxel_features_split = torch.split(voxel_features, 3, dim=3)[0]
        points_mean = voxel_features_split.sum(dim=2, keepdim=True) / voxel_num_points
        f_cluster = voxel_features_split - points_mean
        y1 = torch.split(voxel_features,1, dim=3)
        coords_split_1 = torch.split(coords, 1, dim=2)
        a = y1[0] - (coords_split_1[3] * self.voxel_x + self.x_offset)
        b = y1[1] - (coords_split_1[2] * self.voxel_y + self.y_offset)
        c = y1[2] - (coords_split_1[1] * self.voxel_z + self.z_offset)
        f_center = torch.cat([a, b, c], dim=-1)
        features = [voxel_features, f_cluster, f_center]
        features = torch.cat(features, dim=-1)

        mask = self.get_paddings_indicator(voxel_num_points, voxel_count_data, axis=0)
        features_new = features * mask
        return features_new

class VFETemplate(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre = PillarVFE_pre()
        num_filters = [10, 64]

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, voxel_features, voxel_num_points, coords,voxel_count_data):
        features = self.pre(voxel_features, voxel_num_points, coords,voxel_count_data)

        # torch.Size([1, 4512, 32, 10])
        for pfn in self.pfn_layers:
            features = pfn(features)
        # torch.Size([4512, 64])
        return features

class PointPillarScatter(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_bev_features = 64
        self.nx, self.ny, = 432, 496

    def forward(self, pillar_features, coords):
        if pillar_features.device.type == 'cpu':
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            this_coords = coords
            indices = this_coords[:, :, 1] + this_coords[:, :, 2] * self.nx + this_coords[:, :, 3]
            indices = indices.type(torch.long).squeeze()
            pillars = pillar_features
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            spatial_feature = spatial_feature.view(1, self.num_bev_features, self.ny, self.nx)
            return spatial_feature
        else:
            pillar_features = pillar_features.reshape(1,4512,64,1)
            pillar_features = pillar_features.transpose(1,2)
            out = torch.ops.torch_mlu.plugin_op(coords, pillar_features, self.nx, self.ny)
            return out

class BaseBEVBackbone(nn.Module):
    def __init__(self, input_channels = 64):
        super().__init__()
        layer_nums = [3, 5, 5]
        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        num_upsample_filters = [128, 128, 128]
        upsample_strides = [1, 2, 4]
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.conv_cls = nn.Conv2d(384, 6 * 3, kernel_size=1)
        self.conv_box = nn.Conv2d(384, 6 * 7, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(384, 6 * 2, kernel_size=1)

    # 将box旋转角度限制在0到pi之间 
    def limit_period(self, val, offset=0.5):
        ans = val - torch.floor(val / np.pi + offset) * np.pi 
        return ans

    # 解码anchor
    def decode_torch(self, box_encodings, anchors):
        xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt, rt = torch.split(box_encodings, 1, dim=-1)
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        rg = rt + ra

        out = torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)
        return out

    def generate_predicted_boxes(self, box_preds, dir_cls_preds, anchors):
        dir_cls_preds = dir_cls_preds.reshape(
            dir_cls_preds.shape[0], anchors.shape[1],anchors.shape[2], -1)
        batch_box_preds = box_preds.reshape(anchors.shape)
        batch_box_preds = self.decode_torch(batch_box_preds, anchors) # (1,248,1296,7)

        dir_offset = 0.78539
        dir_limit_offset = 0.0
        dir_labels = torch.max(dir_cls_preds, dim=-1, keepdim=True)[1]

        s = torch.split(batch_box_preds, 6, dim=3)
        dir_rot = self.limit_period(
            s[1] - dir_offset, dir_limit_offset
        )
        if box_preds.device.type == 'mlu':
            dir_labels = dir_labels.half()
        x = dir_rot + dir_offset + np.pi * dir_labels # 合并了box旋转角度和预测方向
        batch_box_preds_new = torch.cat([s[0], x], dim=3)
        return batch_box_preds_new

    def forward(self, spatial_features, anchors):
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        dir_cls_preds = self.conv_dir_cls(x)
        cls_preds = cls_preds.permute(0, 2, 3, 1)
        box_preds = box_preds.permute(0, 2, 3, 1)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1)
        batch_box_preds = self.generate_predicted_boxes(
            box_preds, dir_cls_preds, anchors)
        return cls_preds, batch_box_preds #(1,248,216,18) (1,248,216*6,7)

class pointpillar(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = VFETemplate()
        self.scatter = PointPillarScatter()
        self.backbone_2d = BaseBEVBackbone()
    def forward(self, voxel_features, voxel_num_points, coords, voxel_count_data, anchor):
        features = self.vfe(voxel_features, voxel_num_points, coords, voxel_count_data)
        features = self.scatter(features, coords)
        features = self.backbone_2d(features, anchor)
        return features

net = pointpillar()
state_dict = torch.load("pointpillar_7728_new.pth", map_location="cpu")
point_dict = {}
for key in state_dict['model_state'].keys():
    if 'vfe' in key:
        point_dict[key] = state_dict['model_state'][key]
    elif 'backbone_2d' in key:
        point_dict[key] = state_dict['model_state'][key]
    elif 'dense_head' in key:
        point_dict['backbone_2d.' + key[11:]] = state_dict['model_state'][key]

net.load_state_dict(point_dict)
net.eval()
batch_size  = 4512
voxel_features = np.loadtxt("./inputs/voxel_features.txt")
voxel_features = torch.from_numpy(voxel_features.reshape(1, batch_size, 32, 4)).float()
voxel_num_points = np.loadtxt("./inputs/voxel_num_points.txt")
voxel_num_points = torch.from_numpy(voxel_num_points.reshape(1, batch_size, 1, 1)).float()
coords = np.loadtxt("./inputs/coords.txt")
coords = torch.from_numpy(coords.reshape(1, batch_size, 4, 1)).float()
voxel_count_data = np.arange(32,dtype=np.float).reshape((1,-1))
voxel_count_data = np.broadcast_to(voxel_count_data,(batch_size,32))
voxel_count_data = voxel_count_data.reshape(1, batch_size, 32, 1)
voxel_count_data = torch.from_numpy(voxel_count_data).float()

anchors = np.loadtxt("./inputs/anchors.txt")
anchors = torch.from_numpy(anchors.reshape(1, 248, 216*6, 7)).float()

net(voxel_features, voxel_num_points, coords, voxel_count_data, anchors)

