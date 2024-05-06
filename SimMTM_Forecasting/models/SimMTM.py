import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.losses import AutomaticWeightedLoss
from utils.tools import ContrastiveWeight, AggregationRebuild

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Model(nn.Module):
    """
    Transformer with channel independent + SimMTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.configs = configs

        # Embedding
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder
        if self.task_name == 'pretrain':
            # for reconstruction
            self.projection = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

            # for series-wise representation
            self.pooler = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)

            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'finetune':
            self.head = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

    def forecast(self, x_enc, x_mark_enc):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]

        # embedding
        #x_mark_enc = torch.repeat_interleave(x_mark_enc, repeats=n_vars, dim=0)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        enc_out, attns = self.encoder(enc_out) # enc_out: [(bs * n_vars) x seq_len x d_model]

        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1)) # enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.head(enc_out)  # dec_out: [bs x n_vars x pred_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x pred_len x n_vars]

        # de-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def pretrain_reb_agg(self, x_enc, x_mark_enc, mask):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]

        # embedding
        #x_mark_enc = torch.repeat_interleave(x_mark_enc, repeats=n_vars, dim=0)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        # point-wise
        enc_out, attns = self.encoder(enc_out) # enc_out: [(bs * n_vars) x seq_len x d_model]
        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1)) # enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.projection(enc_out) # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]

        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        # pooler
        pooler_out = self.pooler(dec_out) # pooler_out: [bs x dimension]

        # dec_out for reconstruction / pooler_out for contrastive
        return dec_out, pooler_out

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.unsqueeze(-1) # x_enc: [bs x n_vars x seq_len x 1]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]

        # embedding
        enc_out = self.enc_embedding(x_enc) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        # point-wise representation
        p_enc_out, attns = self.encoder(enc_out) # p_enc_out: [(bs * n_vars) x seq_len x d_model]

        # series-wise representation
        s_enc_out = self.pooler(p_enc_out) # s_enc_out: [(bs * n_vars) x dimension]

        # series weight learning
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [(bs * n_vars) x seq_len x d_model]

        agg_enc_out = agg_enc_out.reshape(bs, n_vars, seq_len, -1) # agg_enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.projection(agg_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]

        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        pred_batch_x = dec_out[:batch_x.shape[0]]

        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())

        # loss
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x

    def forward(self, x_enc, x_mark_enc, batch_x=None, mask=None):

        if self.task_name == 'pretrain':
            return self.pretrain(x_enc, x_mark_enc, batch_x, mask)
        if self.task_name == 'finetune':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
