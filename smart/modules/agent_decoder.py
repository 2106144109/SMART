import pickle
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math
import os


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class SMARTAgentDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_data: Dict,
                 token_size=512) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim)

        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.token_size = token_size
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        self.trajectory_token = token_data['token']
        self.trajectory_token_traj = token_data['traj']
        self.trajectory_token_all = token_data['token_all']
        self.apply(weight_init)
        # 适配数据集时间步：海事默认 shift=1，其它保持 5（不改变默认核心逻辑）
        if isinstance(self.dataset, str) and self.dataset.lower() == 'maritime':
            self.shift = 1
        else:
            self.shift = 5
        self.beam_size = 5
        self.hist_mask = True

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel

    def agent_token_embedding(self, data, agent_category, agent_token_index, pos_a, head_vector_a, inference=False):
        num_agent, num_step, traj_dim = pos_a.shape
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)
        cyc_mask = (agent_type == 2)
        ped_mask = (agent_type == 1)
        # 适配 token 库键：优先 veh/ped/cyc，若不存在则回退 ship
        token_dict = self.trajectory_token
        key_veh = 'veh' if ('veh' in token_dict) else ('ship' if ('ship' in token_dict) else None)
        key_ped = 'ped' if ('ped' in token_dict) else key_veh
        key_cyc = 'cyc' if ('cyc' in token_dict) else key_veh
        trajectory_token_veh = torch.from_numpy(token_dict[key_veh]).clone().to(pos_a.device).to(torch.float) if key_veh else torch.zeros((self.token_size, 2, 3), device=pos_a.device)
        self.agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh.view(trajectory_token_veh.shape[0], -1))
        trajectory_token_ped = torch.from_numpy(token_dict[key_ped]).clone().to(pos_a.device).to(torch.float) if key_ped else trajectory_token_veh
        self.agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped.view(trajectory_token_ped.shape[0], -1))
        trajectory_token_cyc = torch.from_numpy(token_dict[key_cyc]).clone().to(pos_a.device).to(torch.float) if key_cyc else trajectory_token_veh
        self.agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1))

        if inference:
            agent_token_traj_all = torch.zeros((num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device)
            token_all = self.trajectory_token_all
            key_veh_all = 'veh' if ('veh' in token_all) else ('ship' if ('ship' in token_all) else None)
            key_ped_all = 'ped' if ('ped' in token_all) else key_veh_all
            key_cyc_all = 'cyc' if ('cyc' in token_all) else key_veh_all
            trajectory_token_all_veh = torch.from_numpy(token_all[key_veh_all]).clone().to(pos_a.device).to(torch.float) if key_veh_all else torch.zeros((self.token_size, self.shift, 4, 2), device=pos_a.device)
            trajectory_token_all_ped = torch.from_numpy(token_all[key_ped_all]).clone().to(pos_a.device).to(torch.float) if key_ped_all else trajectory_token_all_veh
            trajectory_token_all_cyc = torch.from_numpy(token_all[key_cyc_all]).clone().to(pos_a.device).to(torch.float) if key_cyc_all else trajectory_token_all_veh
            agent_token_traj_all[veh_mask] = torch.cat([trajectory_token_all_veh[:, :self.shift], trajectory_token_veh[:, None, ...]], dim=1)
            agent_token_traj_all[ped_mask] = torch.cat([trajectory_token_all_ped[:, :self.shift], trajectory_token_ped[:, None, ...]], dim=1)
            agent_token_traj_all[cyc_mask] = torch.cat([trajectory_token_all_cyc[:, :self.shift], trajectory_token_cyc[:, None, ...]], dim=1)

        agent_token_emb = torch.zeros(
            (num_agent, num_step, self.hidden_dim),
            device=pos_a.device,
            dtype=self.agent_token_emb_veh.dtype  # align dtype with embeddings under AMP
        )
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        agent_token_traj = torch.zeros((num_agent, num_step, self.token_size, 4, 2), device=pos_a.device)
        agent_token_traj[veh_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyc_mask] = trajectory_token_cyc

        vel = data['agent']['token_velocity']

        categorical_embs = [
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=num_step,
                                                                            dim=0),

            self.shape_emb(data['agent']['shape'][:, self.num_historical_steps - 1, :]).repeat_interleave(
                repeats=num_step,
                dim=0)
        ]
        feature_a = torch.stack(
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
             ], dim=-1)

        x_a = self.x_a_emb(continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
                           categorical_embs=categorical_embs)
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs
        else:
            return feat_a, agent_token_traj

    def agent_predict_next(self, data, agent_category, feat_a):
        num_agent, num_step, traj_dim = data['agent']['token_pos'].shape
        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        token_res = torch.zeros((num_agent, num_step, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])
        return token_res

    def agent_predict_next_inf(self, data, agent_category, feat_a):
        num_agent, traj_dim = feat_a.shape
        agent_type = data['agent']['type']

        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3

        token_res = torch.zeros((num_agent, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])

        return token_res

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None):
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        # 训练阶段可随机屏蔽部分历史时刻以正则化图密度
        if self.hist_mask and self.training:
            rand_cols = torch.randint(0, hist_mask.shape[1], (num_agent, 10), device=hist_mask.device)
            hist_mask[torch.arange(hist_mask.shape[0], device=hist_mask.device).unsqueeze(1), rand_cols] = False

        # 源/目的掩码：推理阶段用 inference_mask 限制目的端；否则都用 hist_mask
        src_mask = hist_mask
        dst_mask = inference_mask if inference_mask is not None else hist_mask

        # 显式构建“同一 agent 内部”的时间边，并映射到展平索引
        num_step = pos_a.shape[1]
        device = pos_a.device
        max_dt = int(self.time_span / self.shift) if self.time_span is not None else num_step - 1
        max_dt = max(1, min(max_dt, num_step - 1))

        time_idx = torch.arange(num_step, device=device)
        agent_idx = torch.arange(num_agent, device=device)

        src_list = []
        dst_list = []
        base = agent_idx[:, None] * num_step  # [A,1]
        for dt in range(1, max_dt + 1):
            t0 = time_idx[:-dt]  # [L]
            t1 = time_idx[dt:]   # [L]
            # 合法的源/目的对（逐 agent）
            valid_pairs = src_mask[:, t0] & dst_mask[:, t1]  # [A,L]
            if valid_pairs.any():
                src_flat = (base + t0[None, :])[valid_pairs]
                dst_flat = (base + t1[None, :])[valid_pairs]
                src_list.append(src_flat)
                dst_list.append(dst_flat)

        if len(src_list) == 0:
            edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)
            r_t = torch.empty((0, 4), dtype=pos_t.dtype, device=device)
            return edge_index_t, r_t

        edge_index_t = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)  # [2, E]

        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(self, data, num_step, agent_category, pos_a, head_a, head_vector_a, mask,
                             batch_s, batch_pl):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                 batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        # 仅在 XY 上计算相对位置，避免 pos_pl(3) 与 pos_s(8) 维度不匹配
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0], :2] - pos_s[edge_index_pl2a[1], :2]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_a = data['agent']['token_pos']
        head_a = data['agent']['token_heading']
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx']
        feat_a, agent_token_traj = self.agent_token_embedding(data, agent_category, agent_token_index,
                                                              pos_a, head_vector_a)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        # agent_valid_mask[~eval_mask] = False
        mask = agent_valid_mask
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(num_step)], dim=0)
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                  for t in range(num_step)], dim=0)
        else:
            batch_s = torch.arange(num_step,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(num_step,
                                    device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])

        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)
        mask[agent_category != 3] = False
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, agent_category, pos_a, head_a,
                                                            head_vector_a, mask, batch_s, batch_pl)

        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(-1, num_step,
                                    self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            # 海事无图或无有效边时跳过 pt2a，保持健壮
            if ('x_pt' in map_enc) and (edge_index_pl2a.shape[1] > 0):
                feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                    repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                        -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        num_agent, num_step, hidden_dim, traj_num, traj_dim = agent_token_traj.shape
        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        next_token_eval_mask[:, -1] = False
        # 仅在历史边界之后产生监督：最早从 t = H-1 开始参与分类
        next_token_eval_mask[:, : self.num_historical_steps - 1] = False

        return {'x_a': feat_a,
                'next_token_idx': next_token_idx,
                'next_token_prob': next_token_prob,
                'next_token_idx_gt': next_token_index_gt,
                'next_token_eval_mask': next_token_eval_mask,
                }

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        pos_a = data['agent']['token_pos'].clone()
        head_a = data['agent']['token_heading'].clone()
        num_agent, num_step, traj_dim = pos_a.shape
        pos_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        head_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift:] = True
        agent_valid_mask[~eval_mask] = False
        agent_token_index = data['agent']['token_idx']
        agent_category = data['agent']['category']
        feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs = self.agent_token_embedding(
            data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True)
        
        agent_type = data["agent"]["type"]
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        av_mask = data["agent"]["av_index"]
        steps_total = data['agent']['position'].shape[1] if ('position' in data['agent']) else \
                      (data['agent']['x'].shape[1] if ('x' in data['agent']) else num_step)
        self.num_recurrent_steps_val = steps_total - self.num_historical_steps
        pred_traj = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val, 2, device=feat_a.device)
        pred_head = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val, device=feat_a.device)
        pred_prob = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val // self.shift, device=feat_a.device)
        next_token_idx_list = []
        mask = agent_valid_mask.clone()
        feat_a_t_dict = {}
        for t in range(self.num_recurrent_steps_val // self.shift):
            if t == 0:
                inference_mask = mask.clone()
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t:] = False
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True
            edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)
            if isinstance(data, Batch):
                batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                     for t in range(num_step)], dim=0)
                batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                      for t in range(num_step)], dim=0)
            else:
                batch_s = torch.arange(num_step,
                                       device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
                batch_pl = torch.arange(num_step,
                                        device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])
            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, agent_category, pos_a, head_a,
                                                                head_vector_a,
                                                                inference_mask, batch_s,
                                                                batch_pl)
            mask_s = inference_mask.transpose(0, 1).reshape(-1)
            edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a,
                                                                batch_s, mask_s)

            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                feat_a = feat_a.reshape(-1, self.hidden_dim)
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
                feat_a = feat_a.reshape(-1, num_step,
                                        self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                if ('x_pt' in map_enc) and (edge_index_pl2a.shape[1] > 0):
                    feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                        repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                            -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

                if i+1 not in feat_a_t_dict:
                    feat_a_t_dict[i+1] = feat_a
                else:
                    feat_a_t_dict[i+1][:, (self.num_historical_steps - 1) // self.shift - 1 + t] = feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]

            next_token_prob = self.token_predict_head(feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t])

            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)

            topk_prob, next_token_idx = torch.topk(next_token_prob_softmax, k=self.beam_size, dim=-1)
            # 保存一份候选索引用于调试打印（在 gather 变成单一索引之前）
            topk_idx_snapshot = next_token_idx.clone()

            expanded_index = next_token_idx[..., None, None, None].expand(-1, -1, self.shift + 1, 4, 2)
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index)

            # 旋转角适配：默认保持 head-based 与 R(-theta)；可通过环境变量启用差分角与偏置/正向旋转
            skip_rot = os.getenv('DECODER_SKIP_ROT', '0') == '1'
            if skip_rot:
                # token 已在 F_local，无需旋转，直接平移到基点
                agent_diff_rel = next_token_traj
            else:
                # 保留原有旋转逻辑（use_diff / ROT_SIGN / ROT_OFFSET）
                use_diff = os.getenv('DECODER_USE_DIFF_ANGLE', '0') == '1'
                if use_diff:
                    # 位置差分角：使用最近两步位移向量的方向作为旋转角
                    base = (self.num_historical_steps - 1) // self.shift
                    if t == 0:
                        idx_cur = base - 1
                        idx_prev = base - 2
                    else:
                        idx_cur = base + t - 1
                        idx_prev = base + t - 2
                    delta = pos_a[:, idx_cur, :2] - pos_a[:, idx_prev, :2]
                    theta = torch.atan2(delta[:, 1], delta[:, 0])
                else:
                    theta = head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
                rot_off = math.radians(float(os.getenv('DECODER_ROT_OFFSET_DEG', '0')))
                theta = theta + rot_off
                cos, sin = theta.cos(), theta.sin()
                rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
                if os.getenv('DECODER_ROT_SIGN', 'neg').lower() == 'pos':
                    rot_mat[:, 0, 0] = cos; rot_mat[:, 0, 1] = -sin
                    rot_mat[:, 1, 0] = sin; rot_mat[:, 1, 1] = cos
                else:
                    rot_mat[:, 0, 0] = cos; rot_mat[:, 0, 1] = sin
                    rot_mat[:, 1, 0] = -sin; rot_mat[:, 1, 1] = cos
                agent_diff_rel = torch.bmm(next_token_traj.view(-1, 4, 2),
                                            rot_mat[:, None, None, ...].repeat(1, self.beam_size, self.shift + 1, 1, 1).view(
                                                -1, 2, 2)).view(num_agent, self.beam_size, self.shift + 1, 4, 2)

            # 后续保持不变
            base_xy = pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, :2]
            agent_pred_rel = agent_diff_rel + base_xy[:, None, None, None, :]

            # 取样策略：默认保持采样；可通过 DECODER_GREEDY=1 改为贪心
            if os.getenv('DECODER_GREEDY', '0') == '1':
                sample_index = topk_prob.argmax(dim=-1, keepdim=True)
            else:
                sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device)
            agent_pred_rel = agent_pred_rel.gather(dim=1,
                                                   index=sample_index[..., None, None, None].expand(-1, -1, self.shift + 1, 4,
                                                                                                    2))[:, 0, ...]
            pred_prob[:, t] = topk_prob.gather(dim=-1, index=sample_index)[:, 0]
            pred_traj[:, t * self.shift:(t + 1) * self.shift] = agent_pred_rel[:, 1:, ...].clone().mean(dim=2)
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_head[:, t * self.shift:(t + 1) * self.shift] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

            pos_a[:, (self.num_historical_steps - 1) // self.shift + t, :2] = agent_pred_rel[:, -1, ...].clone().mean(dim=1)
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            head_a[:, (self.num_historical_steps - 1) // self.shift + t] = theta
            next_token_idx = next_token_idx.gather(dim=1, index=sample_index)
            next_token_idx = next_token_idx.squeeze(-1)
            next_token_idx_list.append(next_token_idx[:, None])

            # 调试打印：观察每步所选 token 与轨迹步进（需设置 DECODER_DEBUG=1）
            _dbg = os.getenv('DECODER_DEBUG', '0') == '1'
            if _dbg and pos_a.shape[0] > 0:
                a0 = 0
                # 当前步位移与前一步位移（用于判断是否ABAB反向交替、步长稳定性）
                base = (self.num_historical_steps - 1) // self.shift
                # cur: 本步写入后的位移（t=0 使用 hist_end 与 hist_end-1）
                cur = pos_a[a0, base + t, :2] - pos_a[a0, base + t - 1, :2]
                if t == 0:
                    prev = pos_a[a0, base - 1, :2] - pos_a[a0, base - 2, :2]
                else:
                    prev = pos_a[a0, base + t - 1, :2] - pos_a[a0, base + t - 2, :2]
                denom = (cur.norm() * prev.norm()).clamp_min(1e-6)
                cos_prev = float((cur @ prev) / denom)
                step_len = float(cur.norm())
                top2 = topk_idx_snapshot[a0, :2].tolist()
                picked = int(next_token_idx[a0].item())
                print(f"[DECODER_DEBUG] t={t:02d} picked={picked} top2={top2} cos(prev,cur)={cos_prev:.3f} step_len={step_len:.2f}")
                # 追加打印 top-5 概率（若 beam_size>=5）
                if topk_prob.shape[0] > 0 and topk_prob.shape[1] >= 5:
                    probs5 = topk_prob[0, :5].detach().cpu().tolist()
                    probs5_str = ",".join([f"{p:.3f}" for p in probs5])
                    print(f"[DECODER_DEBUG] t={t:02d} top5_prob=[{probs5_str}]")
            agent_token_emb[veh_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_veh[
                next_token_idx[veh_mask]]
            agent_token_emb[ped_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_ped[
                next_token_idx[ped_mask]]
            agent_token_emb[cyc_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_cyc[
                next_token_idx[cyc_mask]]
            motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                         pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            vel = motion_vector_a.clone() / (0.1 * self.shift)
            vel[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0
            motion_vector_a[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])], dim=-1)

            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)),
                               categorical_embs=categorical_embs)
            x_a = x_a.view(-1, num_step, self.hidden_dim)

            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        agent_valid_mask[agent_category != 3] = False

        return {
            'pos_a': pos_a[:, (self.num_historical_steps - 1) // self.shift:],
            'head_a': head_a[:, (self.num_historical_steps - 1) // self.shift:],
            'gt': (data['agent']['position'][:, self.num_historical_steps:, :self.input_dim].contiguous()
                   if ('position' in data['agent']) else
                   data['agent']['x'][:, self.num_historical_steps:, :self.input_dim].contiguous()),
            'valid_mask': agent_valid_mask[:, self.num_historical_steps:],
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': torch.cat(next_token_idx_list, dim=-1),
            'next_token_idx_gt': agent_token_index.roll(shifts=-1, dims=1),
            'next_token_eval_mask': data['agent']['agent_valid_mask'],
            'pred_prob': pred_prob,
            'vel': vel
        }