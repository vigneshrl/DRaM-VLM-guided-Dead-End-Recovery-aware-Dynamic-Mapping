import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np

class PointNetEncoder(nn.Module): #pointnet++
    def __init__(self, input_channels=3, output_channels=1024, num_points=4096):
        super(PointNetEncoder, self).__init__()
        self.num_points = num_points
        
        # T-Net for input transform (3x3)
        self.input_transform_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Droupout(0.2), #dropout layer
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Droupout(0.3), #dropout layer
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.input_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels)
        )
        
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), #dropout layer
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), #dropout layer
            nn.Conv1d(128, output_channels // 2, 1),
            nn.BatchNorm1d(output_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2), #Dropout layer 
            nn.Conv1d(output_channels // 2, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        
        # Global features
        self.global_features = nn.Sequential(
            nn.AdaptiveMaxPool1d(1)
        )
        
    def forward(self, x):
        # Random point sampling if needed
        if x.size(2) > self.num_points:
            idx = torch.randperm(x.size(2))[:self.num_points]
            x = x[:, :, idx]
        elif x.size(2) < self.num_points:
            # Pad with duplicated points
            idx = torch.randint(0, x.size(2), (self.num_points - x.size(2),))
            padding = x[:, :, idx]
            x = torch.cat([x, padding], dim=2)
            
        batch_size = x.size(0)
        
        # Input transform
        trans_feat = self.input_transform_net(x)
        trans_feat = trans_feat.view(batch_size, -1)
        trans_mat = self.input_fc(trans_feat).view(batch_size, 3, 3)
        
        # Apply transform
        x = torch.bmm(x.transpose(1, 2), trans_mat).transpose(1, 2)
        
        # Feature extraction
        point_features = self.feature_extraction(x)
        
        # Global features
        global_features = self.global_features(point_features)
        global_features = global_features.view(batch_size, -1)
        
        return global_features, point_features

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super(ImageEncoder, self).__init__()
        # Use ResNet50 as the backbone
        # resnet = resnet50(pretrained=True)
        resnet = resnet50(weights="IMAGENET1K_V1") #mobileVIT 
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add a projection layer to match the output dimension
        self.projection = nn.Conv2d(2048, output_dim, kernel_size=1)
        
    def forward(self, x):
        # Extract features using ResNet
        features = self.backbone(x)  # Shape: [B, 2048, H/32, W/32]
        
        # Project features to the desired dimension
        features = self.projection(features)  # Shape: [B, output_dim, H/32, W/32]
        
        # Get global features through pooling
        global_features = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
        
        return global_features, features

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for all heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(query)  # [B, Lq, E]
        k = self.k_proj(key)    # [B, Lk, E]
        v = self.v_proj(value)  # [B, Lv, E]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lq, D]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, D]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lv, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, Lq, Lk]
        
        # Apply mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, Lq, Lk]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [B, H, Lq, D]
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [B, Lq, E]
        output = self.out_proj(context)  # [B, Lq, E]
        
        return output, attn_weights

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=4, dropout=0.2):
        super(CrossModalFusion, self).__init__()
        
        # Image to LiDAR attention
        self.img_to_lidar_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        
        # LiDAR to Image attention
        self.lidar_to_img_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward layers
        self.ff_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2), #increased to 0.1
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.ff_lidar = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2), #increased to 0.1
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
    def forward(self, img_feats, lidar_feats):
        # Img -> LiDAR attention
        img_lidar_attn, _ = self.img_to_lidar_attn(lidar_feats, img_feats, img_feats)
        lidar_feats = self.norm1(lidar_feats + img_lidar_attn)
        
        # LiDAR -> Img attention
        lidar_img_attn, _ = self.lidar_to_img_attn(img_feats, lidar_feats, lidar_feats)
        img_feats = self.norm2(img_feats + lidar_img_attn)
        
        # Feed-forward for image features
        img_ff = self.ff_img(img_feats)
        img_feats = self.norm3(img_feats + img_ff)
        
        # Feed-forward for LiDAR features
        lidar_ff = self.ff_lidar(lidar_feats)
        lidar_feats = self.norm4(lidar_feats + lidar_ff)
        
        return img_feats, lidar_feats

class DeadEndDetectionModel(nn.Module):
    def __init__(self, img_embed_dim=1024, lidar_embed_dim=1024, fusion_dim=1024, num_heads=4):
        super(DeadEndDetectionModel, self).__init__()
        
        # Image encoders (one for each camera)
        self.front_img_encoder = ImageEncoder(output_dim=img_embed_dim)
        self.right_img_encoder = ImageEncoder(output_dim=img_embed_dim)
        self.left_img_encoder = ImageEncoder(output_dim=img_embed_dim)
        
        # LiDAR encoders (one for each view)
        self.front_lidar_encoder = PointNetEncoder(output_channels=lidar_embed_dim)
        self.right_lidar_encoder = PointNetEncoder(output_channels=lidar_embed_dim)
        self.left_lidar_encoder = PointNetEncoder(output_channels=lidar_embed_dim)
        
        # Projection layers to align dimensions for fusion
        self.img_projection = nn.Linear(img_embed_dim, fusion_dim)
        self.lidar_projection = nn.Linear(lidar_embed_dim, fusion_dim)
        
        # Cross-modal fusion modules (one for each view)
        self.front_fusion = CrossModalFusion(fusion_dim, num_heads)
        self.right_fusion = CrossModalFusion(fusion_dim, num_heads)
        self.left_fusion = CrossModalFusion(fusion_dim, num_heads)
        
        # Final integration layer
        self.integration = nn.Sequential(
            nn.Linear(fusion_dim * 6, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        
        # Output heads
        # 1. Path status classification
        self.path_classifier = nn.Linear(fusion_dim, 3)  # front_open, left_open, right_open #use an MLP
        
        #
        # 2. Dead end classification
        # self.dead_end_classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, 512),
        #     nn.LayerNorm(512),  # Normalize before dropout
        #     nn.ReLU(),
        #     nn.Dropout(0.2),    # Reduced dropout
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 1)
        # )
        self.dead_end_classifier = nn.Sequential(
        nn.Linear(fusion_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
        )
        # )  # is_dead_end
        # for layer in self.dead_end_classifier:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        #         nn.init.constant_(layer.bias, 0.01)
        
        # 3. Direction vectors for open paths (could be angles or vectors)
        self.direction_regressor = nn.Linear(fusion_dim, 9)  # 3 directions × 3 coordinates (x,y,z)
        
        # 4. Confidence scores
        # self.confidence_scorer = nn.Linear(fusion_dim, 3)  # confidence for front, left, right
        
        # In the __init__ method of DeadEndDetectionModel:
        self.spatial_attention = nn.Sequential(
            nn.Linear(img_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 49)  # For 7x7 feature maps, adjust as needed
        )
        self.lidar_spatial_attention = nn.Sequential(
            nn.Linear(lidar_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4096)  # For 64x64 feature maps, adjust as needed
        )
        
    def apply_spatial_attention(self, global_feat, spatial_feat):
        B, C, H, W = spatial_feat.shape
        spatial_flat = spatial_feat.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        spatial_attn = F.softmax(self.spatial_attention(global_feat).view(B, 1, -1), dim=-1)
        attended_feats = torch.bmm(spatial_attn, spatial_flat).squeeze(1)
        enhanced_feat = global_feat + attended_feats
        return enhanced_feat
    
    def apply_spatial_attention_lidar(self, global_feat, point_feat):
        B, C, N = point_feat.shape
        point_flat = point_feat.transpose(1, 2)  # [B, N, C]
        point_attn = F.softmax(self.lidar_spatial_attention(global_feat).view(B, 1, -1)[:,:,:N], dim=-1)
        attended_feats = torch.bmm(point_attn, point_flat).squeeze(1)
        enhanced_feat = global_feat + attended_feats
        return enhanced_feat
        
    def forward(self, front_img, right_img, left_img, front_lidar, right_lidar, left_lidar):
        # Process images
        front_img_global, front_img_feats = self.front_img_encoder(front_img)
        right_img_global, right_img_feats = self.right_img_encoder(right_img)
        left_img_global, left_img_feats = self.left_img_encoder(left_img)
        
        # Process LiDAR
        front_lidar_global, front_lidar_feats = self.front_lidar_encoder(front_lidar)
        right_lidar_global, right_lidar_feats = self.right_lidar_encoder(right_lidar)
        left_lidar_global, left_lidar_feats = self.left_lidar_encoder(left_lidar)
        
        # Process and enhance images
        front_img_enhanced = self.apply_spatial_attention(front_img_global, front_img_feats)
        right_img_enhanced = self.apply_spatial_attention(right_img_global, right_img_feats)
        left_img_enhanced = self.apply_spatial_attention(left_img_global, left_img_feats)
        
        # Apply similar attention to LiDAR features (if useful for your task)
        # For LiDAR data, you'd need to adapt the approach based on your feature dimensions
        front_lidar_enhanced = self.apply_spatial_attention_lidar(front_lidar_global, front_lidar_feats)
        right_lidar_enhanced = self.apply_spatial_attention_lidar(right_lidar_global, right_lidar_feats)
        left_lidar_enhanced = self.apply_spatial_attention_lidar(left_lidar_global, left_lidar_feats)
        
        # Project enhanced features to common dimension
        front_img_proj = self.img_projection(front_img_enhanced)
        right_img_proj = self.img_projection(right_img_enhanced)
        left_img_proj = self.img_projection(left_img_enhanced)
        
        # For LiDAR, use either enhanced features (if implemented) or original global features
        front_lidar_proj = self.lidar_projection(front_lidar_enhanced)
        right_lidar_proj = self.lidar_projection(right_lidar_enhanced)
        left_lidar_proj = self.lidar_projection(left_lidar_enhanced)
        
        # Reshape for attention
        batch_size = front_img.size(0)
        front_img_proj = front_img_proj.view(batch_size, 1, -1)
        right_img_proj = right_img_proj.view(batch_size, 1, -1)
        left_img_proj = left_img_proj.view(batch_size, 1, -1)
        
        front_lidar_proj = front_lidar_proj.view(batch_size, 1, -1)
        right_lidar_proj = right_lidar_proj.view(batch_size, 1, -1)
        left_lidar_proj = left_lidar_proj.view(batch_size, 1, -1)
        
        # Apply cross-modal fusion
        front_img_fused, front_lidar_fused = self.front_fusion(front_img_proj, front_lidar_proj)
        right_img_fused, right_lidar_fused = self.right_fusion(right_img_proj, right_lidar_proj)
        left_img_fused, left_lidar_fused = self.left_fusion(left_img_proj, left_lidar_proj)
        
        # Concatenate all fused features
        all_fused = torch.cat([
            front_img_fused.squeeze(1), front_lidar_fused.squeeze(1),
            right_img_fused.squeeze(1), right_lidar_fused.squeeze(1),
            left_img_fused.squeeze(1), left_lidar_fused.squeeze(1)
        ], dim=1)
        
        # Integrate all features
        integrated_feats = self.integration(all_fused)

        #path logits 
        path_logits = self.path_classifier(integrated_feats)
        path_probs = torch.sigmoid(path_logits)

       #dead end logits
        dead_end_logits = self.dead_end_classifier(integrated_feats)  # [B, 1]
        # direction_vectors = self.direction_regressor(integrated_feats).view(batch_size, 3, 3)  # [B, 3, 3]
        # any_path_open = (path_probs > 0.5).any(dim=1, keepdim=True)
        # regulated_dead_end = torch.where(
        #     any_path_open,
        #     torch.tensor(-5.0, device=path_probs.device),  # Sigmoid(-100) ≈ 0
        #     dead_end_logits
        # )
        # Output predictions
        # path_status = torch.sigmoid(self.path_classifier(integrated_feats))  # [front_open, left_open, right_open]
        # is_any_path_open = torch.max(path_status, dim=1, keepdim=True)[0]
        # is_dead_end = 1.0 - is_any_path_open  # [is_dead_end]
        # # learned_dead_end = torch.sigmoid(self.dead_end_classifier(integrated_feats))  # [is_dead_end]
        # final_dead_end = (is_dead_end + dead_end_logits) / 2.0  # [is_dead_end]

        direction_vectors = self.direction_regressor(integrated_feats).view(batch_size, 3, 3)  # 3 directions × [x,y,z]
        # confidence_scores = torch.sigmoid(self.confidence_scorer(integrated_feats))  # [front_conf, left_conf, right_conf]
        
        return {
            'path_status': path_probs,  # Binary: Is path open in each direction?
            'is_dead_end': torch.sigmoid(torch.sigmoid(dead_end_logits)),  # Binary: Is this a dead end?
            'path_logits': path_logits,  # Regression: Direction vectors for each open path
            'dead_end_logits': dead_end_logits,  
            'direction_vectors': direction_vectors  # Regression: Direction vectors for each open path
            # 'confidence_scores': confidence_scores  # Regression: Confidence in each direction
        }