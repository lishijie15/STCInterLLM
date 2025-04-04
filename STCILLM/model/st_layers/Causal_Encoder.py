import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.fft
from transformers.configuration_utils import PretrainedConfig
from STCILLM.model.st_layers.ModernTCN import extendmodel

def block_index_to_coordinates(row_index): 
    if not 0 <= row_index < 1177:
        raise ValueError("块索引必须在0到1177之间")
    
    block_size = 1

    top_left_y1 = row_index * block_size
    top_left_y2 = row_index * block_size

    return (top_left_y1, top_left_y2)

def mixup_region(graph_array, x, y, block_size):
    # print(graph_array.shape)
    h, w, _ = graph_array.shape
    x_start = max(x - block_size, 0)
    x_end = min(x + 2*block_size, h)
    y_start = max(y - block_size, 0)
    y_end = min(y + 2*block_size, w)
    # print(x_start, x_end, y_start, y_end)
    if y_end > graph_array.shape[1] - block_size:  
        block_size = 1
    
    region = graph_array[:, y:y+block_size, :]
    surrounding_regions = []

    for j in range(int(y_start / block_size), int(y_end / block_size)):
        # print(i, j)
        surrounding_region = graph_array[:, j*block_size:(j+1)*block_size, :]
        surrounding_regions.append(surrounding_region)

    mixed_region = 0
    for j in range(len(surrounding_regions)):
        mixed_region += surrounding_regions[j] / len(surrounding_regions)

    
    new_graph_array = graph_array.clone()
    new_graph_array[:, y:y + block_size, :] = mixed_region.to(torch.uint8)
    # graph_array[:, y:y+block_size, x:x+block_size, :] = mixed_region.to(torch.uint8)
    return new_graph_array

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,blmc->blnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('blni,io->blno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class Spatial_Attention(nn.Module):
    def __init__(self, out_features):
        super(Spatial_Attention, self).__init__()
        self.out_features = out_features

        self.xff = nn.Linear(self.out_features, 3 * self.out_features)

        self.ff = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.GELU(),
            nn.Linear(self.out_features, self.out_features),
        )

        self.ln = nn.LayerNorm(self.out_features)
        # self.ln1 = nn.LayerNorm(self.out_features)
    def forward(self, input):
        x_ = self.xff(input)

        x_ = torch.stack(torch.split(x_, self.out_features, -1), 0)
        query = x_[0]
        key = x_[1]
        value = x_[2]

        # Wh1 = torch.matmul(query, self.a[:self.out_features, :])
        # Wh2 = torch.matmul(key, self.a[self.out_features:, :])
        # broadcast add
        # e = Wh1 + Wh2.T

        e = torch.matmul(query, key.transpose(-1, -2)) / (self.out_features ** 0.5)
        attn = torch.softmax(e, -1)

        self.attention_maps = attn.detach()
        processed_images = []
        for batch_attention_maps, graph_array in zip(self.attention_maps, value):
            # 
            graph_array_front = graph_array[:12, :, :] 
            graph_array_back = graph_array[12:, :, :]  
            # print(batch_attention_maps.shape)
            min_number = 3
            
            column_sums = torch.sum(batch_attention_maps, dim=1)
            _, min_weight_rows = torch.topk(column_sums, min_number, dim=1, largest=False)
            # min_weight_columns = min_weight_columns.reshape(batch_attention_maps.shape[0], -1)
            # print(min_weight_columns)

            # 
            for column_index_all, i in zip(min_weight_rows, range(len(min_weight_rows))):
                for column_index in column_index_all:
                    # x, y = block_index_to_coordinates(column_index)
                    x, y = i, column_index
                    # print(y.dtype)
                    graph_array = mixup_region(graph_array_front, x, y, 1)  # 
            #
            processed_image = torch.cat((graph_array, graph_array_back), dim=0)
            # 
            processed_images.append(processed_image)

        new_value = torch.stack(processed_images, dim=0)

        
        value_i = torch.matmul(attn, new_value)

        value = self.ff(value_i) + input
        return self.ln(value)

class ST_Enc(nn.Module):
    def __init__(self, args):
        super(ST_Enc, self).__init__()
        self.config = PretrainedConfig()
        self.num_nodes = args.num_nodes
        self.feature_dim = args.input_dim

        self.input_window = args.input_window
        self.output_window = args.output_window
        self.output_dim = args.output_dim

        self.dropout = 0.
        self.dilation_exponential = 1

        self.conv_channels = args.conv_channels
        self.residual_channels = args.residual_channels
        self.skip_channels = args.skip_channels
        self.end_channels = args.end_channels
        # self.extend_channels = args.extend_channels

        self.layers = 3
        self.propalpha = 0.05

        self.plus_window = self.input_window + self.output_window
        self.plus_proj = nn.Linear(self.input_window, self.plus_window)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.extend = extendmodel(args)
        self.d_model = args.d_model
        self.sattn = Spatial_Attention(self.residual_channels)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(DilatedInception(self.residual_channels,
                                                          self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.residual_channels,
                                                        self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))
                if self.plus_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.plus_window-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        if self.plus_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.plus_window), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.plus_window-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)


    def forward(self, source):
        inputs = source
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
        inputs = self.plus_proj(inputs)

        if self.plus_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

        x = self.extend(inputs)

        x = self.sattn(x.transpose(1, 3)).transpose(1, 3)

        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))  
        for i in range(self.layers):
            residual = x
            filters = self.filter_convs[i](x)   
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)    
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)   
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x_emb = x.clone()
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, x_emb

