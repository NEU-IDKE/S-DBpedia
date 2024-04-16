import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .Model import Model

class DistMult_GDR(Model):
	def __init__(self, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None , in_path = './', use_gpu = True):
		super(DistMult_GDR, self).__init__(ent_tot, rel_tot)

		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		
		self.use_gpu = use_gpu
		self.coor_dic = self.load_coor(in_path)
	
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h, t, r, mode):
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t
		score = torch.sum(score, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		
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
		return (score, dis)

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score, _ = self.forward(data)
		score = -score
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