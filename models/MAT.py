import torch
import torch.nn as nn
from models.ir50 import Backbone
from models.vit_model import VisionTransformer
from thop import profile
from models.MLMAT_funcs import *
import os.path
from models.facial_landmarks.SLPT import Sparse_alignment_network
from models.facial_landmarks.Config.default import _C as cfg


class SEblock(nn.Module):

    def __init__(self, channel, r=0.5):
        super(SEblock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # tianjia  FAvg(·) and FMax(·) denote the operation of AdaptiveAvgPool and AdaptiveMaxPool
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):

        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)
        weight = self.fc(branch)

        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))
        # tianjia _max
        branch_max = self.global_max_pool(x)
        branch_max = branch_max.view(branch_max.size(0), -1)

        weight_max = self.fc(branch_max)

        h1, w1 = weight_max.shape
        weight_max = torch.reshape(weight_max,(h1, w1, 1, 1))

        # scale = weight * x + weight_max * x
        weight = weight + weight_max
        # mean = x.mean(dim=[0,1])
        return weight


class get_box(nn.Module):

    def __init__(self, num_points, half_length, img_size):
        super(get_box, self).__init__()
        self.img_size = img_size
        self.num_points = num_points
        self.half_length = torch.tensor([[[half_length, half_length]]], dtype=torch.float32)
        self.half_length.requires_grad = False

    def forward(self, anchor):
        Bs = anchor.size(0)
        half_length = (self.half_length.to(anchor.device) / (self.img_size)).repeat(Bs, 1, 1)
        bounding_min = torch.clamp(anchor - half_length, 0.0, 1.0)
        bounding_max = torch.clamp(anchor + half_length, 0.0, 1.0)
        bounding_box = torch.cat((bounding_min, bounding_max), dim=2)

        bounding_xs = torch.nn.functional.interpolate(bounding_box[:, :, 0::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_ys = torch.nn.functional.interpolate(bounding_box[:, :, 1::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_xs, bounding_ys = bounding_xs.unsqueeze(3).repeat_interleave(self.num_points, dim=3), \
                                   bounding_ys.unsqueeze(2).repeat_interleave(self.num_points, dim=2)

        meshgrid = torch.stack([bounding_xs, bounding_ys], dim=-1)
        return meshgrid  # bounding_box,


class patches_generator(nn.Module):
    def __init__(self):
        super(patches_generator, self).__init__()

    def forward(self, feature_maps, meshgrid):

        ROI_features = []
        for i in range(98):
            feature_map_samples = torch.nn.functional.grid_sample(feature_maps, meshgrid[:, i, :, :, :],
                                                                  mode='bilinear', padding_mode='border',
                                                                  align_corners=None)
            ROI_features.append(feature_map_samples)

        ROI_features = torch.stack(ROI_features)
        ROI_features_out = ROI_features.permute(1, 0, 2, 3, 4)

        return ROI_features_out


class MAT_ir50(nn.Module):
    def __init__(self, img_size=224, num_classes=7, dims=[64, 128, 256], embed_dim=768, drop_rate=0):
        super().__init__()

        self.img_size = img_size
        self.num_point = 98
        self.Sample_num = 7

        self.dims = dims
        self.embed_dim = embed_dim
        # self.num_heads = num_heads
        # self.dim_head = []
        # for num_head, dim in zip(num_heads, dims):
        #     self.dim_head.append(int(torch.div(dim, num_head).item()))
        self.num_classes = num_classes

        self.VIT = VisionTransformer(depth=2, embed_dim=embed_dim)

        self.face_land = Sparse_alignment_network(98, 256, 8, 1024, './models/pretrain/init_98.npz', cfg)
        face_land_checkpoint = torch.load(os.path.join('./models/pretrain', 'WFLW_6_layer.pth'))
        self.face_land = load_pretrained_weights(self.face_land, face_land_checkpoint)

        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)

        self.patches_generator = patches_generator()

        self.ROI_1 = get_box(self.Sample_num, 2.0, 28)  # num_points, half_length, img_size
        self.ROI_2 = get_box(self.Sample_num, 2.0, 14)
        self.ROI_3 = get_box(self.Sample_num, 2.0, 7)

        '''
        self.ROI_1 = get_box(14, GAR_DR.0, 112)  # num_points, half_length, img_size
        self.ROI_2 = get_box(28, GAR_DR.0, 112)
        self.ROI_3 = get_box(56, GAR_DR.0, 112)
        '''
        self.arrangement_0 = nn.PixelShuffle(4)
        self.arrangement_1 = nn.PixelShuffle(8)
        self.arrangement_2 = nn.PixelShuffle(16)

        self.arm = Amend_raf()
        self.fc = nn.Linear(121, num_classes)  # 169

        self.branch_SE = SEblock(channel=7)
        self.drop_rate = drop_rate

        # structure encoding
        self.structure_encoding = nn.Parameter(torch.randn(1, 98, embed_dim))

        # self.structure_encoding = nn.Parameter(torch.randn(1, num_points, d_model))

        self.feature_extractor_1 = nn.Sequential(nn.Conv2d(self.dims[0], embed_dim, kernel_size=3, stride=2, padding=1),
                                                nn.Conv2d(embed_dim, embed_dim, kernel_size=4,
                                                            bias=False))

        self.feature_extractor_2 = nn.Sequential(nn.Conv2d(self.dims[1], embed_dim, kernel_size=3, stride=2, padding=1),
                                                nn.Conv2d(embed_dim, embed_dim, kernel_size=4,
                                                            bias=False))

        self.feature_extractor_3 = nn.Sequential(nn.Conv2d(self.dims[2], embed_dim, kernel_size=3, stride=2, padding=1),
                                                nn.Conv2d(embed_dim, embed_dim, kernel_size=4,
                                                            bias=False))

        # self. feature_extractor_3_1 = nn.Conv2d(98, 98, kernel_size=3, stride=GAR_DR, padding=1)
        # self.fc_back = nn.Linear(512, num_classes)
        '''
        self.linear_layer_1 = nn.Linear(in_features=784, out_features=768)
        self.linear_layer_2 = nn.Linear(in_features=1568, out_features=768)
        self.feature_extractor_3_1 = nn.Conv2d(98, 98, kernel_size=3, stride=GAR_DR, padding=1)
        self.linear_layer_3_2 = nn.Linear(in_features=784, out_features=768)
        '''
    def forward(self, x):
        bs = x.size(0)

        ##########################################################
        # --------Multi-level feature pre-extractor (MFE)---------
        x_ir1, x_ir2, x_ir3, x_ir4, x_ir50 = self.ir_back(x)
        '''
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x_ir50)
        x = x.view(x.size(0), -1)
        IR_out = self.fc_back(x_ir50)

        '''
        x_ir1, x_ir2, x_ir3 = self.conv1(x_ir1), self.conv2(x_ir2), self.conv3(x_ir3)

        ##########################################################
        # --------Global amending attention (GAA)---------
        x_arm_0 = self.arrangement_0(x_ir1)
        x_arm_1 = self.arrangement_1(x_ir2)
        x_arm_2 = self.arrangement_2(x_ir3)
        x_arm = torch.cat([x_arm_0, x_arm_1, x_arm_2], dim=1)
        x_arm_SE = self.branch_SE(x_arm)
        # A = x_arm + x_arm_SE
        x_arm_out, alpha= self.arm(x_arm * x_arm_SE) # , alpha

        if self.drop_rate > 0:
            x_arm_out = nn.Dropout(self.drop_rate)(x_arm_out)

        x_out = x_arm_out.view(x_arm_out.size(0), -1)

        GAA_out = self.fc(x_out)

        ##########################################################
        # --------Local patch transformer (LPT) ---------

        facial_lands = self.face_land(x)
        initial_lands = facial_lands[:, -1, :, :]

        meshgrid_1 = self.ROI_1(initial_lands.detach())  # ROI_1,
        meshgrid_2 = self.ROI_2(initial_lands.detach())  # ROI_2,
        meshgrid_3 = self.ROI_3(initial_lands.detach())  # ROI_3,

        '''
        ROI_1features = self.patches_generator(x_arm_0, meshgrid_1)  # 28
        ROI_2features = self.patches_generator(x_arm_1, meshgrid_2)  # 14
        ROI_3features = self.patches_generator(x_arm_2, meshgrid_3)  # 7
        
        # ROI_feature_1 = ROI_1features.reshape(bs, self.num_point, 28, 28)
        # o1 = self.feature_extractor_1(ROI_feature_1).view(bs, self.num_point, self.embed_dim)
        o1 = ROI_1features.view(bs, self.num_point, -1)
        # ROI_feature_1 = ROI_1features.reshape(-1, self.dims[0], self.Sample_num, self.Sample_num)
        o1 = self.linear_layer_1(o1)

        ROI_feature_2 = ROI_2features.view(bs, self.num_point, -1)
        
        #  ROI_feature_2 = ROI_2features.reshape(bs, self.num_point, self.Sample_num, self.Sample_num)
        # o2 = self.feature_extractor_2(ROI_feature_2).view(bs, self.num_point, self.embed_dim)
        o2 = self.linear_layer_2(ROI_feature_2)
        # ROI_feature_3 = ROI_3features.reshape(bs, self.num_point, 28, 28)
        ROI_feature_3 = ROI_3features.squeeze()

        # o3 = self.feature_extractor_3(ROI_feature_3).view(bs, self.num_point, self.embed_dim)
        o3 = self.feature_extractor_3_1(ROI_feature_3).view(bs, self.num_point, -1)
        '''

        ROI_1features = self.patches_generator(x_ir1, meshgrid_1)  #28
        ROI_2features = self.patches_generator(x_ir2, meshgrid_2)  #14
        ROI_3features = self.patches_generator(x_ir3, meshgrid_3)  #7

        ROI_feature_1 = ROI_1features.reshape(-1, self.dims[0], self.Sample_num, self.Sample_num)
        o1 = self.feature_extractor_1(ROI_feature_1).view(bs, self.num_point, self.embed_dim)

        ROI_feature_2 = ROI_2features.reshape(-1, self.dims[1], self.Sample_num, self.Sample_num)
        o2 = self.feature_extractor_2(ROI_feature_2).view(bs, self.num_point, self.embed_dim)

        ROI_feature_3 = ROI_3features.reshape(-1, self.dims[2], self.Sample_num, self.Sample_num)
        o3 = self.feature_extractor_3(ROI_feature_3).view(bs, self.num_point, self.embed_dim)

        o1_s = o1 + self.structure_encoding
        o2_s = o2
        o3_s = o3
        o = torch.cat([o1_s, o2_s, o3_s], dim=1)
        # o_s = o + self.structure_encoding
        
        LPT_out = self.VIT(o)

        # GAA_out, LPT_out
        return GAA_out, LPT_out


class Amend_raf(nn.Module):
    def __init__(self, inplace=7):
        super(Amend_raf, self).__init__()
        self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        mask = torch.tensor([]).cuda()
        # mask = torch.tensor([])
        createVar = locals()
        # N = x.size(1) # N=GAR_DR  i：0，1
        for i in range(x.size(1)):

            createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
            createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
            mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        x = self.bn(mask)

        #self.alpha = nn.Parameter(torch.tensor([1.0]))
        xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        xmin, _ = torch.min(x, 1, keepdim=True)
        x_GAR = xmean + self.alpha * global_mean

        return x_GAR, self.alpha


def compute_param_flop():
    model = MAT_ir50()
    img = torch.rand(size=(1, 3, 224, 224))
    flops, params = profile(model, inputs=(img,))
    print(f'flops:{flops / 1000 ** 3}G,params:{params / 1000 ** 2}M')


if __name__ == '__main__':
    model = MAT_ir50(img_size=224, num_classes=7)
    input = torch.randn(64, 3, 224, 224)
    output = model(input)
    print(output.size())

