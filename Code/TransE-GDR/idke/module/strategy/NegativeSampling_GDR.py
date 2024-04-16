import torch
from .Strategy import Strategy

class NegativeSampling_GDR(Strategy):
	'''
	loss:
	batch_size:
	regul_rate: ???
	l3_regul_rate: ???
	
	'''
	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling_GDR, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate  
		self.l3_regul_rate = l3_regul_rate  

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]  # 正样本
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)  # 维度转换[0,1]-->[1,0]
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:] # 负样本
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		score, dis = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)

		p_dis = self._get_positive_dis(dis)
		n_dis = self._get_negative_dis(dis)

		num = 1
		w = 1 / (torch.abs(torch.log10((p_dis + num)/(n_dis + num))) + 1)
		n_score = w * n_score
		
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res

	def _get_positive_dis(self, dis):
		positive_dis = dis[:self.batch_size]  # 正样本
		positive_dis = positive_dis.view(-1, self.batch_size).permute(1, 0)  # 维度转换[0,1]-->[1,0]
		return positive_dis


	
	def _get_negative_dis(self, dis):
		negative_dis = dis[self.batch_size:] # 负样本
		negative_dis = negative_dis.view(-1, self.batch_size).permute(1, 0)
		return negative_dis
