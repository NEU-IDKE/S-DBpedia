import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from .Model import Model


class TransE_GDR(Model):
	'''
	p_norm:  L1/L2
	norm_flag: 是否用范数
	margin: ???
	epsilon: ???
	'''
	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, in_path = './',use_gpu = True):
		super(TransE_GDR, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon  # 这是啥
		self.norm_flag = norm_flag  # 这是啥标志位
		self.p_norm = p_norm  # 范数  
		self.use_gpu = use_gpu
        
		self.coor_dic = self.load_coor(in_path)
        


		# 实体、关系 嵌入
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		# ???
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

		
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)  # -1???
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		# normal 和 head_batch 具体区别 ???
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		
		
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()  #
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
		# if self.margin_flag:
		# 	return self.margin - score
		# else:
		# 	return score
		return (score, dis)

	# 这是啥玩意
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

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
