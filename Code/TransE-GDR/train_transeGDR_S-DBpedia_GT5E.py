import openke
from openke.config import Trainer, Tester
from openke.module.loss import MarginLoss

from openke.data import TrainDataLoader, TestDataLoader

import idke
from idke.module.model import TransE_GDR
from idke.module.strategy import NegativeSampling_GDR

import os
import argparse


path = "../openke-data/benchmarks/S-DBpedia_GT5E/"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = path, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,  
	neg_rel = 0) 

# dataloader for test
test_dataloader = TestDataLoader(path, "link")

# define the model                           
transe_gdr = TransE_GDR(
	in_path = path + "coor.txt",
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True
    )


# define the loss function
model = NegativeSampling_GDR(
	model = transe_gdr, 
	loss = MarginLoss(margin = 10),
	batch_size = train_dataloader.get_batch_size()
    )

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1, use_gpu = True)
trainer.run()
transe_gdr.save_checkpoint(log_path + 'transe_gdr-S-DBpedia_GT5E.ckpt')

# test the model
transe_gdr.load_checkpoint(log_path + 'transe_gdr-S-DBpedia_GT5E.ckpt')
tester = Tester(model = transe_gdr, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

