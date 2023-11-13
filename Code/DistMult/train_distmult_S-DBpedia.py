import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../openke-data/benchmarks/S-DBpedia/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("../openke-data/benchmarks/S-DBpedia/", "link")

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.05
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1, use_gpu = True, opt_method = "adagrad")
trainer.run()
distmult.save_checkpoint('../openke-data/checkpoint/S-DBpedia_distmult.ckpt')

# test the model
distmult.load_checkpoint('../openke-data/checkpoint/S-DBpedia_distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
