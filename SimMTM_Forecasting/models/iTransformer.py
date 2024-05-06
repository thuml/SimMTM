import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from utils.losses import AutomaticWeightedLoss
from utils.tools import ContrastiveWeight, AggregationRebuild

class Flatten_Head(nn.Module):
    def __init__(self, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-1)
        self.linear = nn.Linear(d_model, pred_len, bias=True)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x n_vars x d_model]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x   # x: [bs x n_vars x seq_len]

class Pooler_Head(nn.Module):
    def __init__(self, nf, dimension=128, head_dropout=0):
        super().__init__()

        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(nf, nf // 2),
            nn.BatchNorm1d(nf // 2),
            nn.ReLU(),
            nn.Linear(nf // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [bs x n_vars x d_model]
        x = self.pooler(x) # [bs x dimension]
        return x

class Model(nn.Module):
    """
    iTransformer + SimMTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.configs = configs

        # patching and embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

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

            # for series-wise representation
            self.pooler = Pooler_Head(configs.enc_in*configs.d_model, head_dropout=configs.head_dropout)

            # for reconstruction
            self.projection = Flatten_Head(configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'finetune':
            self.head = Flatten_Head(configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
    
    def forecast(self, x_enc, x_mark_enc):

        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape   # x_enc: [Batch Time Variate]

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # encoder
        enc_out, _ = self.encoder(enc_out)

        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N]

        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):

        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # encoder
        enc_out = self.enc_embedding(x_enc)
        p_enc_out, _ = self.encoder(enc_out)  # p_enc_out: [bs x n_vars x d_model]

        # series-wise representation
        s_enc_out = self.pooler(p_enc_out) # s_enc_out: [bs x dimension]

        # series weight learning
        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [bs x bs]
        rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [bs x n_vars x d_model]

        # decoder
        dec_out = self.projection(agg_enc_out)  # [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs x seq_len x n_vars]

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
