import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from .Model import Model


'''
rand_init: ???
'''
class TransR_GDR(Model):
         
    def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, rand_init = False, margin = None, in_path = './', use_gpu = True):
        super(TransR_GDR, self).__init__(ent_tot, rel_tot)
        
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.rand_init = rand_init
        self.use_gpu = use_gpu

        self.coor_dic = self.load_coor(in_path)


        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)

        # 这一步操作是要干什么 ? 
        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_r)
            for i in range(min(self.dim_e, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_e)
            for i in range(self.rel_tot):
                self.transfer_matrix.weight.data[i] = identity
        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score
    
    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
            e = torch.matmul(e, r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.dim_e)
            e = torch.matmul(e, r_transfer)
        return e.view(-1, self.dim_r)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.transfer_matrix(batch_r)
        h = self._transfer(h, r_transfer)
        t = self._transfer(t, r_transfer)

        h_ids = batch_h.to('cpu').numpy()
        t_ids = batch_t.to('cpu').numpy()
        h_coor = np.array([[float(self.coor_dic[h_id][0]), float(self.coor_dic[h_id][1])] for h_id in h_ids]) 
        t_coor = np.array([[float(self.coor_dic[t_id][0]), float(self.coor_dic[t_id][1])] for t_id in t_ids])
        if self.use_gpu:
            h_coor = torch.tensor(h_coor).to(torch.device('cuda'))
            t_coor = torch.tensor(t_coor).to(torch.device('cuda'))
        else:
            h_coor = torch.tensor(h_coor).to(torch.device('cpu'))
            t_coor = torch.tensor(t_coor).to(torch.device('cpu'))
        coor = h_coor - t_coor
        dis = torch.norm(coor, p=2, dim=1)


        score = self._calc(h ,t, r, mode)
        # if self.margin_flag:
        #     return self.margin - score
        # else:
        #     return score
        return (score, dis)

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.transfer_matrix(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2) +
                 torch.mean(r_transfer ** 2)) / 4
        return regul * regul

    def predict(self, data):
        score, _ = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def load_coor(self,in_path):
        raw_data = pd.read_csv(in_path,
                               sep='\t',
                               header=None,
                               names=['ent', 'longitude', 'latitude'],
                               keep_default_na=False,
                               encoding='utf-8')
        
        raw_data = raw_data.applymap(lambda x: str(x).strip()) #去除首位空格 
        #以字典的键值对形式存储，其中元素作为key，其计数作为value
        coor_dic = dict([(int(triple[0]), [triple[1], triple[2]]) for triple in raw_data.values])
        return coor_dic 
