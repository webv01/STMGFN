import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from torch import einsum, nn
import pickle
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False,TrAten=True,feed_forward_dim=256,dropout=0.0):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.TrAten=TrAten


        self.FC_Q = nn.Conv2d(model_dim, model_dim, kernel_size=1, bias=True)
        self.FC_K = nn.Conv2d(model_dim, model_dim, kernel_size=1, bias=True)
        if TrAten:
            self.FC_V = nn.Conv2d(in_channels=model_dim,
                                                   out_channels=model_dim,
                                                   kernel_size=(1, 1), dilation=2)
            self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
             nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        else:
            self.FC_V = nn.Linear(model_dim, model_dim)
            self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
             nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
            # self.feed_forward = FeedForwardGNN(model_dim, dropout=dropout)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,dim=-2):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        residual_out=x
        x = x.transpose(dim, -2)
        query,key,value=x,x,x

        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]


        query = self.FC_Q(query.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        key = self.FC_K(key.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        if self.TrAten:
            value = self.FC_V(value.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        else:
            value = self.FC_V(value)



        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)



        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)
        out = out.transpose(dim, -2)
        out = self.dropout1(out)
        out = self.ln1(residual_out + out)
        if self.TrAten:
            residual = out
            out = self.feed_forward(out)
            out = self.dropout2(out)
            out = self.ln2(residual + out)
            return out
        else:
            residual = out
            out = self.feed_forward(out)
            out = self.dropout2(out)
            out = self.ln2(residual + out)
            return out

class ST_block(nn.Module):
    def __init__(
   self,  model_dim,feed_forward_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        # self.PositionAttention=PositionAttention(12)
        # self.ChannelAttention=ChannelAttention()
        self.model_dim=model_dim
        self.attn_layers_t = AttentionLayer(model_dim, num_heads,TrAten=True
                ,feed_forward_dim=feed_forward_dim,dropout=dropout)
        self.attn_layers_s = AttentionLayer(model_dim, num_heads,TrAten=False
                ,feed_forward_dim=feed_forward_dim,dropout=dropout)

        # self.GatedFusion=GatedFusion(model_dim)
    def forward(self, x):

        out = self.attn_layers_t(x, dim=1)

        out = self.attn_layers_s(out, dim=2)



        return out

class nconv(nn.Module):
    """Graph conv operation."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("nm,bmc->bnc", A, x)

        return x.contiguous()
class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):

        x = torch.einsum('bvc,bvw->bwc', (x, A))
        return x.contiguous()

class linear(nn.Module):
    """Linear layer."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in, c_out, dropout, support_len=4, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        # self.gconv_RNN=gconv_RNN()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.weights = nn.Parameter(torch.FloatTensor(c_in, c_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, support):
        out = [x]
        for a in support:

            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2
#         for a in support[-2:]:

#             x1 = self.gconv_RNN(x, a.to(x.device))
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = self.gconv_RNN(x1, a.to(x.device))
#                 out.append(x2)
#                 x1 = x2
        h = torch.cat(out, dim=-1)
        h = torch.einsum('bni,io->bno', h, self.weights) + self.bias
        # h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support
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
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, support_len=4, order=2 ):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = gcn(dim_in+self.hidden_dim, 2*dim_out, dropout=0, support_len=support_len, order=order )
        self.update = gcn(dim_in+self.hidden_dim, dim_out, dropout=0, support_len=support_len, order=order )

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out,  num_layers, support_len=4, order=2):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out,  support_len=support_len, order=order))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, support_len=support_len, order=order))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden

class STMGFN(nn.Module):
    def __init__(
        self,
        num_nodes,
        adj_mx=None,
        DTW=None,
        mem_num=30,
        top_k=10,
        alpha=5,
        rnn_units=96,
        support_len=4,
        order=2,
        in_steps=36,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=32,
        tod_embedding_dim=16,
        dow_embedding_dim=16,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=32,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        use_mixed_proj=True,
    ):
        super().__init__()


        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        # self.posi=TemporalPositionalEncoding(input_embedding_dim,dropout)
        self.adj_mx=adj_mx
        self.DTW=DTW
        self.rnn_units=rnn_units

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.top_k=top_k
        


        self.input_proj = nn.Linear(1, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
            self.time_step_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(out_steps, num_nodes, adaptive_embedding_dim))
            )
        # self.x_input_proj = nn.Linear(self.model_dim ,self.rnn_units)
        # self.model_dim=self.rnn_units





        self.attn_layers = nn.ModuleList(
            [
                ST_block(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers2 = nn.ModuleList(
            [
                ST_block(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers3 = nn.ModuleList(
            [
                ST_block(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(self.out_steps, 1, kernel_size=(1, 1))
                for _ in range(num_layers)
            ]
        )
        self.tran_conv = nn.ModuleList(
            [
                nn.Conv2d(self.model_dim, self.rnn_units, kernel_size=(1, 1))
                for _ in range(num_layers)
            ]
        )

        self.decoder_num_layers=1
        self.output_dim=output_dim
        self.decoder = nn.ModuleList(
            [
                ADCRNN_Decoder(self.num_nodes, self.model_dim, self.rnn_units,
                               self.decoder_num_layers, support_len=support_len, order=order)
                for _ in range(num_layers)
            ]
        )



        # self.proj = nn.Sequential(
        #     nn.Linear(self.rnn_units, self.model_dim,  bias=True),
        #
        # )
        self.out_proj =nn.Sequential(
            nn.Linear(self.rnn_units,self.output_dim,  bias=True),
            # nn.GELU(),
            # nn.Linear(feed_forward_dim, self.output_dim,  bias=True),

        )
        self.time_conv2 = nn.Conv2d(in_steps, out_steps, kernel_size=(1, 1))
        self.mem_num=mem_num
        self.mem_dim=rnn_units
        self.memory = self.construct_memory()
        # self.alpha=alpha
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(alpha)
        # self.GCN1_tg=gcn(self.rnn_units, self.adaptive_embedding_dim, dropout=0, support_len=support_len-2, order=1 )
        # self.GCN2_tg=gcn(self.rnn_units, self.adaptive_embedding_dim, dropout=0, support_len=support_len-2, order=1 )

    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['MK'].t()), dim=-1)
        att_score=att_score/torch.sum(att_score,dim=-1,keepdim=True)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['MV'])
        
        # _, ind = torch.topk(att_score, k=2, dim=-1)
        # pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        # neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        return value+h_t #, query, pos, neg
    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        # memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, 32), requires_grad=True)
        memory_dict['MV'] = nn.Parameter(torch.randn(self.mem_num, self.rnn_units), requires_grad=True)
        memory_dict['MK'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
       
        memory_dict['We1'] = nn.Parameter(torch.randn(self.model_dim,self.adaptive_embedding_dim), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.model_dim,self.adaptive_embedding_dim), requires_grad=True) # project memory to embedding
        
        memory_dict['node1'] = nn.Parameter(torch.randn(self.num_nodes, self.adaptive_embedding_dim), requires_grad=True)
        memory_dict['node2'] = nn.Parameter(torch.randn(self.num_nodes, self.adaptive_embedding_dim), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    def preprocessing(self, adj):
        adj = adj + torch.eye(self.num_nodes).to(adj.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return adj
    def graph_constructor(self,node_embeddings1, node_embeddings2,sim):
        nodevec1 = torch.tanh(self.alpha*node_embeddings1)
        nodevec2 = torch.tanh(self.alpha*node_embeddings2)

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        
#         nodevec1 = torch.tanh(node_embeddings1)
#         nodevec2 = torch.tanh(node_embeddings2)


#         adj = F.relu(torch.tanh(self.alpha*torch.mm(nodevec1, nodevec2.T)))


        mask = torch.zeros_like(adj)
        mask.fill_(float('0'))
        s1, t1 = sim.topk( self.top_k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = torch.mul(adj, mask)
        return F.relu(torch.tanh(self.alpha*adj))
    def sim_global(self,flow_data, sim_type='att'):
        """Calculate the global similarity of traffic flow data.
        :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
        :param type: str, type of similarity, attention or cosine. ['att', 'cos']
        :return sim: tensor, symmetric similarity, [v,v]
        """

        if len(flow_data.shape) == 4:
            

            
            n,l,v,c = flow_data.shape
            att_scaling = n * l * c
            cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1 # cal 2-norm of each node, dim N
            sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
            scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
            sim = sim * scaling
        elif len(flow_data.shape) == 3:

            n,v,c = flow_data.shape
            att_scaling = n * c
            cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1 # cal 2-norm of each node, dim N
            sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
            scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
            sim = sim * scaling


        return sim
    def pivotalconstruct(self, x, adj, k):#torch.Size([64, 1, 75, 13])
        x = x.squeeze(1)#torch.Size([64, 75, 13])
        x = x.sum(dim=0)#torch.Size([75, 13])
        y = x.sum(dim=1).unsqueeze(0)
        adjp = torch.einsum('ij, jk->ik', x[:,:-1], x.transpose(0, 1)[1:,:]) / y
        adjp = adjp * adj
        score = adjp.sum(dim=0) + adjp.sum(dim=1)
        N = x.size(0)
        _, topk_indices = torch.topk(score,k)
        mask = torch.zeros(N, dtype=torch.bool,device=x.device)
        mask[topk_indices] = True
        masked_matrix = adjp * mask.unsqueeze(1) * mask.unsqueeze(0)
        adjp = F.softmax(F.relu(masked_matrix), dim=1)
        return adjp
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        support_set=[]
        support_set.extend(self.adj_mx)
        support_set.extend(self.DTW)
        support_set.append([])
        support_set.append([])     
        support_set.append([]) 
 



        
        
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1] #tod.shape#Out[9]: torch.Size([16, 12, 170])
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x[..., : 1])  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        # x=self.posi(x.transpose(1,2)).transpose(1,2)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                (dow * 7).long()

            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        X_adp_emb=None
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
            X_adp_emb=self.adaptive_embedding

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        # x=self.x_input_proj(x)
        # node_graph = torch.matmul(self.memory['node1'], self.memory['node2'].T)
        node_embeddings1 = torch.matmul(self.memory['node1'], self.memory['We1'].T)
        node_embeddings2 = torch.matmul(self.memory['node2'], self.memory['We2'].T)
        sim_global_DA=self.sim_global(x[:,:self.out_steps,::])
        graph_1=self.graph_constructor(self.memory['node1'],self.memory['node2'],sim_global_DA)
        sim_global_DA=self.sim_global(x[:,self.out_steps:self.out_steps*2,::])
        graph_2=self.graph_constructor(self.memory['node1'],self.memory['node2'],sim_global_DA)
        sim_global_DA=self.sim_global(x[:,self.out_steps*2:,::])
        graph_3=self.graph_constructor(self.memory['node1'],self.memory['node2'],sim_global_DA)
        
        
        # graph_1=self.pivotalconstruct(x[:,:self.out_steps,::][..., : 1].transpose(1,-1),node_graph,self.top_k)
        # graph_2=self.pivotalconstruct(x[:,self.out_steps:self.out_steps*2,::][..., : 1].transpose(1,-1),node_graph,self.top_k)
        # graph_3=self.pivotalconstruct(x[:,self.out_steps*2:,::][..., : 1].transpose(1,-1),node_graph,self.top_k)
        support_set[-1]=self.preprocessing(graph_1)
        support_set[-2]=self.preprocessing(graph_2)
        support_set[-3]=self.preprocessing(graph_3)
        support_set.append([])

        encoder_x=[]


        for i in range(0,self.num_layers):
            # x= self.attn_layers[i](x)
            x_1= self.attn_layers[i](x[:,:self.out_steps,::])
            x_2= self.attn_layers2[i](x[:,self.out_steps:self.out_steps*2,::])
            x_3= self.attn_layers3[i](x[:,self.out_steps*2:,::])
            x = torch.cat([x_1,x_2,x_3], dim=1)
            x=self.time_conv2(x).squeeze(1) 


            encoder_x.append(self.conv[i](self.tran_conv[i](x.transpose(1,-1)).transpose(1,-1)).squeeze(1))
        h_t=torch.mean(torch.stack(encoder_x, dim=1), dim=1)
        

        ht_list = [h_t]*1
        out = []
        # dict={}
        # dict["adj_mx"]=support_set[0]
        # dict["DTW"]=support_set[1]
        # dict["graph_1"]=support_set[2]
        # dict["graph_2"]=support_set[3]
        # dict["graph_3"]=support_set[4]

        


        for t in range(self.out_steps):
            node_embeddings1 = torch.matmul(self.time_step_embedding[t,::], self.memory['We1'].T)
            node_embeddings2 = torch.matmul(self.time_step_embedding[t,::], self.memory['We2'].T)
            
            ht_list[0]= self.query_memory(ht_list[0])#去掉增强模块
            sim_global_DA=self.sim_global(ht_list[0])
            DA=self.graph_constructor(node_embeddings1,node_embeddings2,sim_global_DA)
            # dict[t]=DA


            support_set[-1]=self.preprocessing(DA)
            # support_set[-2]=self.preprocessing(DA.transpose(0,1))
            #+x[:, t, ...]
            h_de, ht_list = self.decoder[i](x[:, t, ...], ht_list, support_set)
            out.append(h_de)
        # with open('my_dict.pkl', 'wb') as f:
        #     pickle.dump(dict, f)           
        x = torch.stack(out, dim=1)

        x = self.out_proj(x)



        return x
