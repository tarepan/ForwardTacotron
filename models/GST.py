import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from Hyperparameters import Hyperparameters as hp
import hparams as hp

class GST(nn.Module):

    def __init__(self):

        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL()

    def forward(self, inputs, style_selection=None):
        inputs = inputs.transpose(1,2)
        enc_out = self.encoder(inputs)  # computes reference encoder
        style_embed, attn_scores = self.stl(enc_out)
        # TODO: style token direct selection
        # if style_selection is None:
        #     enc_out = self.encoder(inputs) # computes reference encoder
        #     style_embed, attn_scores = self.stl(enc_out)
        # else:
        #     attn_scores = self.scalars_to_heads(style_selection) # [h, N, 1, num_style_tokens]
        #     keys = torch.tanh(self.stl.embed).unsqueeze(0).expand(-1, -1, -1)  # [N, token_num, E // num_heads]
        #     out = torch.matmul(attn_scores, keys)  # [h, N, T_q, num_units/h]
        #     out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        return style_embed, attn_scores
    
    # def scalars_to_heads(self, scalars: torch.tensor):
    #     # scalars [N, num_style_tokens]
    #     scalars = scalars.repeat(hp.num_heads,1, 1)  #[h, N, num_style_tokens] e.g. [8, 32, 10]
    #     return scalars[:,:,None,:] #[h, N, 1, num_style_tokens] e.g. [8, 32, 1, 10]
    


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.num_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.tts_embed_dims // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self):

        super().__init__()
        # self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.E // hp.num_heads))
        self.embed = nn.Parameter(torch.rand(hp.token_num, hp.tts_embed_dims // hp.num_heads))
        # self.embed = (torch.ones(1,hp.tts_embed_dims // hp.num_heads).transpose(0,1)*torch.arange(1,hp.token_num+1)).transpose(0,1)
        d_q = hp.tts_embed_dims // 2
        d_k = hp.tts_embed_dims // hp.num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.tts_embed_dims, num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, attn_score = self.attention(query, keys)

        return style_embed, attn_score


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        # query  [N, 1, E//2]
        # key  [N, token_num, E // num_heads]
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
        # TODO: decide whether to pass keys (style tokens) to dense or not when selecting

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        # this is scaled dot prod. Authors used content based attention instead
        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.cosine_similarity(querys, keys, dim=-1)[:,:, None, :] # ([8, 32, 1, 10])
        # scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k] [8, 32, 1, 10])
        # scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3) # [h, N, 1, T_k]

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out, scores


class AttentionRNN(nn.Module):
    '''
    input:
        inputs: [N, T_y, E//2]
        memory: [N, T_x, E]
    output:
        attn_weights: [N, T_y, T_x]
        outputs: [N, T_y, E]
        hidden: [1, N, E]
    T_x --- character len
    T_y --- spectrogram len
    '''

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=hp.tts_embed_dims // 2, hidden_size=hp.tts_embed_dims, batch_first=True, bidirectional=False)
        self.W = nn.Linear(in_features=hp.tts_encoder_dims, out_features=hp.tts_encoder_dims, bias=False)
        self.U = nn.Linear(in_features=hp.tts_encoder_dims, out_features=hp.tts_encoder_dims, bias=False)
        self.v = nn.Linear(in_features=hp.tts_encoder_dims, out_features=1, bias=False)

    def forward(self, inputs, memory, prev_hidden=None):
        T_x = memory.size(1)
        T_y = inputs.size(1)

        # inputs = torch.cat([inputs[:, 0, :].unsqueeze(1), inputs[:, :-1, :]], 1)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(inputs, prev_hidden)  # outputs: [N, T_y, E]  hidden: [1, N, E]
        w = self.W(outputs).unsqueeze(2).expand(-1, -1, T_x, -1)  # [N, T_y, T_x, E]
        u = self.U(memory).unsqueeze(1).expand(-1, T_y, -1, -1)  # [N, T_y, T_x, E]
        attn_weights = self.v(F.tanh(w + u).view(-1, hp.tts_encoder_dims)).view(-1, T_y, T_x)
        attn_weights = F.softmax(attn_weights, 2)

        return attn_weights, outputs, hidden
    
if __name__ == '__main__':
    # import numpy as np
    stl = STL()
    ref_e = ReferenceEncoder()
    gst = GST()
    rand_inp = torch.rand(2,80,73)
    out = gst(rand_inp)
    scalars = torch.tensor([1., 0., 0., 0, 0, 0, 0, 0, 0, 0])
    scalars = scalars.repeat(8, 2, 1, 1)
    # new_out = torch.matmul(scalars, values) # values from mhattention
    # new_out = torch.cat(torch.split(new_out, 1, dim=0), dim=3).squeeze(0)
