import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import hparams as hp


class GST(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL()
    
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        enc_out = self.encoder(inputs)  # computes reference encoder
        style_embed, attn_scores = self.stl(enc_out)
        return style_embed, attn_scores
    
    def forward_with_scores(self, scalars, batch_size):
        # scalars = torch.tensor([1., 0., 0., 0, 0, 0, 0, 0, 0, 0])
        scalars = scalars.repeat(hp.num_heads, batch_size, 1, 1)
        style_embed, attn_scores = self.stl.forward_with_scores(attention_scores=scalars, batch_size=batch_size)
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
        #
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
        d_q = hp.tts_embed_dims // 2
        d_k = hp.tts_embed_dims // hp.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.tts_embed_dims,
                                            num_heads=hp.num_heads)
        
        init.normal_(self.embed, mean=0, std=0.5)
    
    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        # TODO: Add cosine similarity loss for tokens instead of tanh
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, attn_score = self.attention(query, keys)
        return style_embed, attn_score
    
    def forward_with_scores(self, attention_scores, batch_size):
        keys = torch.tanh(self.embed).unsqueeze(0).expand(batch_size, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, attn_score = self.attention.forward_with_scores(keys, scores=attention_scores)
        
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
        # TODO: some don't project this
        # values = self.W_value(key)
        # TODO: decide whether to pass keys (style tokens) to dense or not when selecting
        split_size = self.num_units // self.num_heads
        
        querys = self.W_query(query)  # [N, T_q, num_units] (n_units  = hp.tts_embed_dims)
        keys = self.W_key(key)  # [N, T_k, num_units]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        # scores = torch.cosine_similarity(querys, keys, dim=-1)[:, :, None, :]  # ([8, 32, 1, 10])
        # scores = F.softmax(scores, dim=3)  # [h, N, 1, T_k]
        # values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        # TODO: if not project, tile instead
        # vs = tf.tile(tf.expand_dims(v, axis=1), [1, hp.num_heads, 1, 1])
        values = key.unsqueeze(0).repeat(hp.num_heads, 1, 1, 1)
        
        # this is scaled dot prod. Authors used content based attention instead
        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k] [8, 32, 1, 10])
        scores = scores / (self.key_dim ** 0.5)
        
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        return out, scores
    
    def forward_with_scores(self, key, scores):
        # query  [N, 1, E//2]
        # key  [N, token_num, E // num_heads]
        #TODO: try not to project
        # values = key
        # values = self.W_value(key)
        # TODO: decide whether to pass keys (style tokens) to dense or not when selecting
        split_size = self.num_units // self.num_heads
        # values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = key.unsqueeze(0).repeat(hp.num_heads, 1, 1, 1)
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        return out, scores


if __name__ == '__main__':
    stl = STL()
    ref_e = ReferenceEncoder()
    gst = GST()
    rand_inp = torch.rand(2, 80, 73)
    out = gst(rand_inp)
    scalars = torch.tensor([1., 0., 0., 0, 0, 0, 0, 0, 0, 0])
    out_scored = gst.forward_with_scores(scalars, 2)
