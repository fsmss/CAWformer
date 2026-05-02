import torch
import torch.nn as nn
import math
from math import sqrt
from torch import Tensor
import torch.nn.functional as F
from typing import  Optional
import numpy as np
import pywt
import matplotlib.pyplot as plt
def visualize_series(original_series, denoised_series, series_index=0, channel_index=0):
    # 选择某个时间序列和通道进行可视化
    original_data = original_series.permute(0, 2, 1)[series_index, channel_index, :].cpu().detach().numpy()
    denoised_data = denoised_series.permute(0, 2, 1)[series_index, channel_index, :].cpu().detach().numpy()

    # 绘制原始数据和去噪后的数据在同一个图上
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(denoised_data, label='Denoised Data', color='orange', linestyle='--')
    
    plt.title('Original vs Denoised Time Series (Channel {})'.format(channel_index))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png')  # 保存图片
    plt.show()  # 显示图像
class CAWformer(nn.Module):
    def __init__(self,
                 enc_in,
                 seq_len,
                 d_model,
                 dropout,
                 pe_type,
                 kernel_size,
                 n_layers=3,
                 klist=[],
                 rate=0.01
                ):
        
        super(CAWformer, self).__init__()
        self.klist=klist
        self.rate=rate
        self.n_layers=n_layers
        self.LD=LD(kernel_size=kernel_size)
        self.varCor_attn_blocks=nn.ModuleList([
            VarCorAttention_attn_block(enc_in,d_model,dropout)
            for _ in range(self.n_layers)
        ])
        self.auto_attn_blocks=nn.ModuleList([
            auto_attn_block(enc_in,d_model,dropout)
            for _ in range(self.n_layers)
        ])
        self.position_embedder=DataEmbedding(pe_type=pe_type,seq_len=seq_len, 
                                            d_model=d_model,c_in=enc_in) 

    def wavelet_denoising(self, residual):

        residual_np = residual.cpu().detach().numpy()

        coeffs = pywt.wavedec(residual_np, 'db1', axis=-1, level=2,mode='periodization')  #mode='smooth' periodization

        for i in range(1, len(coeffs)):  #2 3
            coeffs[i] = pywt.threshold(coeffs[i], self.rate * np.max(np.abs(coeffs[i])))

        residual_filtered_np = pywt.waverec(coeffs, 'db1', axis=-1)
 
        residual_filtered_np = residual_filtered_np[..., :residual_np.shape[-1]]

        residual_filtered = torch.tensor(residual_filtered_np).to(residual.device)
        return residual_filtered
    def forward(self, inp):
        inp=self.position_embedder(inp.permute(0,2,1)).permute(0,2,1)
        main=self.LD(inp)#趋势项
        residual=inp-main
        residual = self.wavelet_denoising(residual)
        res_1=residual #torch.Size([128, 256, 7])
        res_2=residual.permute(0,2,1)

        for i in range(self.n_layers):
            res_2=self.varCor_attn_blocks[i](res_2)
        for i in range(self.n_layers):
            res_1=self.auto_attn_blocks[i](res_1)            
        res=res_1+res_2.permute(0,2,1)
        return res, main
class VarCorAttention_attn_block(nn.Module):
    def __init__(self,enc_in,d_model,dropout):
        super(VarCorAttention_attn_block, self).__init__()
        self.channel_att_norm=nn.BatchNorm1d(enc_in)
        self.fft_norm=nn.LayerNorm(d_model)
        self.channel_attn=VarCorAttention(None, None,None, attention_dropout=dropout,output_attention=False)
        self.fft_layer = nn.Sequential(
                                nn.Linear(d_model, int(d_model*2)),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(int(d_model*2), d_model),
                                )
    def forward(self, residual):
        res_2=self.channel_att_norm(self.channel_attn(residual,residual,residual)[0]+residual)
        res_2=self.fft_norm(self.fft_layer(res_2)+res_2)
        return res_2  
class VarCorAttention(nn.Module):
    def __init__(self, args,  factor=5, scale=None, attention_dropout=0.1, output_attention=False) -> None:
        super(VarCorAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def optimized_compute_cross_cor(self, queries, keys):
        # dtype = torch.float32  # 或者使用 torch.float16，具体取决于模型的其他部分
        # queries = queries.to(dtype)
        # keys = keys.to(dtype)
        # Perform batched FFT
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        # Expand dimensions for broadcasting
        q_fft = q_fft.unsqueeze(2)  # [B, D, 1, T/2+1]
        k_fft = torch.conj(k_fft.unsqueeze(1))  # [B, 1, D, T/2+1] 
        # Element-wise multiplication and batched inverse FFT
        res = q_fft * k_fft  # [B, D, D, T/2+1] #res[b, i, j, :]：第 b 个批次中，queries 的第 i 个特征与 keys 的第 j 个特征之间的频域乘积。
        corr = torch.fft.irfft(res, dim=-1) #corr[b, i, j, t]：第 b 个批次中，queries[b, i, :] 与 keys[b, j, :] 在时刻 t 的互相关值。
        # Mean across the time dimension
        corr = corr.mean(dim=-1)
        return corr
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, D, T = queries.shape
        _, S, _ = values.shape
        corr = torch.zeros(B, D, D).to(queries.device)
        scale = self.scale or 1./sqrt(T)
        for i in range(B):
            corr[[i], :, :] = self.optimized_compute_cross_cor(queries[[i], :, :], keys[[i], :, :])
        corr = torch.softmax(corr*scale, dim=-1)
        V = torch.einsum("bsd,bde->bse", corr, values)
        if self.output_attention:
            return (V.contiguous(), corr)
        else:
            return (V.contiguous(), None)
class auto_attn_block(nn.Module):
    def __init__(self,enc_in,d_model,dropout):
        super(auto_attn_block, self).__init__()
        self.auto_attn_norm=nn.BatchNorm1d(enc_in)
        self.fft_norm=nn.LayerNorm(d_model)
        self.auto_attn=Auto_Attention(P=64,d_model=d_model,proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
                                nn.Linear(d_model, int(d_model*2)),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(int(d_model*2), d_model),
                                )
    def forward(self, residual):
        res_1=self.auto_attn_norm((self.auto_attn(residual)+residual).permute(0,2,1))
        res_1=self.fft_norm(self.fft_layer(res_1)+res_1)
        return  res_1.permute(0,2,1)


class Auto_Attention(nn.Module):
    def __init__(self, P,d_model,proj_dropout=0.2,kernel_size=25):
        """
        Initialize the Auto-Attention module.

        Args:
            d_model (int): The input and output dimension for queries, keys, and values.
        """
        super(Auto_Attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.P=P
        self.out_projector = nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(proj_dropout))
        self.scale = nn.Parameter(torch.tensor(d_model ** -0.5), requires_grad=False)
        self.kernel_size=25
    def forward(self, inp):
        inp = inp.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]
        T = inp.size(-1)
        cat_sequences = [inp]
        index = int(T / self.P) - 1 if T % self.P == 0 else int(T / self.P)
        for i in range(index):
            end = (i + 1) * int(self.P)
            # Concatenate sequences to support autoregressive behavior
            cat_sequence = torch.cat([inp[:, :, end:], inp[:, :, 0:end]], dim=-1)
            cat_sequences.append(cat_sequence)
        # Stack the concatenated sequences
        output = torch.stack(cat_sequences, dim=-1)
        # Permute the output for attention calculation
        output = output.permute(0, 1, 3, 2)
        # Apply autoregressive self-attention
        output = self.auto_attention(output).squeeze(-2)
        output=self.out_projector(output).permute(0, 2, 1)
        return output   
    def auto_attention(self, inp):
        # Separate query and key
        query = self.W_Q(inp[:, :, 0, :].unsqueeze(-2))  # Query
        keys = self.W_K(inp)  # Keys
        values = self.W_V(inp)  # Values

        # Calculate dot product
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * (self.scale)

        # Normalize attention scores
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attn_scores, values)

        return output

class LD(nn.Module):
    def __init__(self,kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True) 
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        laplace_weights = torch.distributions.Laplace(0.0, 1.0).sample(self.conv.weight.shape)
        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights,dim=-1)

        self.conv.bias.data.fill_(0.0)
        
    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)
        
        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
    



class DataEmbedding(nn.Module):
    def __init__(self, pe_type,seq_len, d_model,c_in,dropout=0.):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)
def SinCosPosEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)

    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None or pe == 'no':
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)
