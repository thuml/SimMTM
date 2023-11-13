import torch
import torch.nn as nn
import numpy as np
import random
import math
import torch.nn.functional as F
  
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
       
    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    
    
class ContrastiveLoss(nn.Module):

    def __init__(self, device, args):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.temperature = args.temperature
        
        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        
    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size, oral_batch_size):
        
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)
        
        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(cur_batch_size//oral_batch_size):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size*i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size*i)
            positives_mask += ll
            positives_mask += lr
        
        positives_mask = torch.from_numpy(positives_mask)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0
        
        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om, batch_x):
        
        cur_batch_shape = batch_emb_om.shape
        oral_batch_shape = batch_x.shape
        
        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
        
        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0], oral_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)
        
        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1) 
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]) / positives.shape[-1],  torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(self.device).float()
        
        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        
        return loss, similarity_matrix, logits
    
   
class RebuildLoss(torch.nn.Module):

    def __init__(self, device, args):
        super(RebuildLoss, self).__init__()
        self.args = args
        self.device = device
        self.temperature = args.temperature
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om, batch_emb_o, batch_x):
        
        cur_batch_shape = batch_emb_om.shape
        oral_batch_shape = batch_x.shape
        
        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature
        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(self.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        
        batch_emb_om = batch_emb_om.view(cur_batch_shape[0], -1)
        
        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb[:oral_batch_shape[0]].reshape(oral_batch_shape[0], cur_batch_shape[1], -1)
        
        # MSE Loss
        if self.args.rbtp == 0:
            loss = self.mse(rebuild_oral_batch_emb, batch_emb_o.detach())
        elif self.args.rbtp == 1:
            loss = self.mse(rebuild_oral_batch_emb, batch_x.detach())
        
        return loss, rebuild_weight_matrix
    
