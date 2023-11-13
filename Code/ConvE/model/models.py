from helper import *

class ConvE(torch.nn.Module):
	def __init__(self, args, num_entities, num_relations):
		super(ConvE, self).__init__()
		self.emb_e = torch.nn.Embedding(num_entities, args.embed_dim, padding_idx=0)
		self.emb_rel = torch.nn.Embedding(num_relations, args.embed_dim, padding_idx=0)
		self.inp_drop = torch.nn.Dropout(args.input_drop)
		self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
		self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
		self.loss = torch.nn.BCELoss()
		self.emb_dim1 = args.k_h
		self.emb_dim2 = args.embed_dim // self.emb_dim1
		self.hidden_size = (self.emb_dim1 * 2 - 2) * (self.emb_dim2 - 2) * 32
		self.conv1 = torch.nn.Conv2d(1, 32, (args.ker_sz, args.ker_sz), 1, 0, bias=args.bias)
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.bn2 = torch.nn.BatchNorm1d(args.embed_dim)
		self.register_parameter('b', Parameter(torch.zeros(num_entities)))
		self.fc = torch.nn.Linear(self.hidden_size, args.embed_dim)
		print(num_entities, num_relations)

	def init(self):
		xavier_normal_(self.emb_e.weight.data)
		xavier_normal_(self.emb_rel.weight.data)

	def forward(self, e1, rel):
		e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
		rel_embedded = self.emb_e(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

		stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

		stacked_inputs = self.bn0(stacked_inputs)
		x = self.inp_drop(stacked_inputs)
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.feature_map_drop(x)
		x = x.view(x.shape[0], -1)
		x = self.fc(x)
		x = self.hidden_drop(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
		x += self.b.expand_as(x)
		pred = torch.sigmoid(x)

		return pred

