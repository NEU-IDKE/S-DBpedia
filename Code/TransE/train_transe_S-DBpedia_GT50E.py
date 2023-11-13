from re import T
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/S-DBpedia_GT50E/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,  # 25
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/S-DBpedia_GT50E/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 10.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe_S-DBpedia_GT50E.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_S-DBpedia_GT50E.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
