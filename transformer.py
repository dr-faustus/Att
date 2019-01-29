import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, emb_dim, max_len, batch_size, word_padding_token):
        super(PositionalEncoder, self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.word_padding_token = word_padding_token

        n_position = max_len + 1

        self.position_enc = torch.nn.Embedding(n_position, emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_dim)
        self.position_enc.weight.requires_grad = False

    def get_absolute_pos(self, word_sequences):
        batch = []
        max_len = word_sequences.size(1)
        batch_size = word_sequences.size(0)
        for word_seq in word_sequences:
            start_idx = 1
            word_pos = []
            for pos in word_seq:
                if pos == self.word_padding_token:
                    word_pos.append(0)
                else:
                    word_pos.append(start_idx)
                    start_idx += 1
            batch.append(torch.from_numpy(np.array(word_pos)).type(torch.LongTensor))

        batch = torch.cat(batch).view(batch_size, max_len)
        return Variable(batch)

    def forward(self, word_seq):
        word_pos = self.get_absolute_pos(word_seq)
        if torch.has_cudnn:
            word_pos = word_pos.cuda()
        word_pos_emb = self.position_enc(word_pos)
        return word_pos_emb


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.T = 1

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output = enc_input
        for i in range(self.T):
            enc_output, enc_slf_attn = self.slf_attn(
                enc_output, enc_output, enc_output, attn_mask=slf_attn_mask, mask=non_pad_mask)
        # if non_pad_mask is not None:
        #     enc_output *= non_pad_mask

        # enc_output = self.pos_ffn(enc_output)
        # if non_pad_mask is not None:
        #     enc_output *= non_pad_mask

        return enc_output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))

        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = self.softmax(attn)
        attn = attn.masked_fill(attn_mask, 0.0)

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.attention_linear = nn.ModuleList([
            nn.Linear(d_v, d_v // 2) for _ in range(n_head)
        ])

        self.input_gate = nn.ModuleList([
            nn.Linear(d_model, d_v) for _ in range(n_head)
        ])
        # self.head_gate = nn.ModuleList([
        #     nn.Linear(d_v, n_head) for _ in range(n_head)
        # ])

        self.fc = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        for i in range(n_head):
            nn.init.xavier_normal_(self.attention_linear[i].weight)

        self.highway = torch.nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q = torch.tanh(q)
        # k = torch.tanh(k)
        # v = torch.tanh(v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # q = squash(q)
        # v = squash(v)
        # k = squash(k)

        # if mask is not None:
        #     mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, attn_mask=torch.cat([attn_mask for i in range(n_head)], dim=0))

        # print(output)
        # exit()

        output = output.view(n_head, sz_b, len_q, d_v)
        gate_value = torch.cat([self.input_gate[i](output[i]).unsqueeze(0) for i in range(n_head)], dim=0)
        gate_value = F.softmax(gate_value, dim=0)
        output = torch.sum(gate_value * output, dim=0)
        # print(output.shape)
        # output = squash(output)

        # output = torch.cat([squash(self.attention_linear[i](output[i])).unsqueeze(0) for i in range(n_head)], dim=0)
        # gate = self.gate(residual)
        #
        # gate = F.softmax(gate, dim=-1)
        # print(gate.shape)
        # print(output.shape)
        # exit()
        # output = sum([output[i] * gate[:, :, i].unsqueeze(-1) for i in range(self.n_head)])

        # output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.fc(output)
        # output = self.dropout(output)
        # output = squash(output)

        # highway_value = F.sigmoid(self.highway(residual))

        # output = highway_value * output + (1 - highway_value) * residual
        output = output + residual
        #
        # output = self.layer_norm(output)
        output = output.masked_fill(mask, 0.0)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        # output = self.w_2(F.relu(self.w_1(output)))
        output = self.w_2(squash(self.w_1(output), dim=1))

        output = output.transpose(1, 2)
        output = self.dropout(output)

        # output = output + residual
        # output = self.layer_norm(output)
        # print(torch.norm(output[0, 0, :], dim=-1))
        # exit()

        return output


class ACTBasic(nn.Module):
    def __init__(self,hidden_size):
        super(ACTBasic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while ((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any():
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if encoder_output:
                state, _ = fn((state, encoder_output))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return previous_state, (remainders,n_updates)