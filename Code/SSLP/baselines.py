from pathlib import Path
from tqdm import tqdm
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import RESCAL, DistMult
# DistMult, ComplEx, RESCAL,\
# ConvE, ConvKB, \ 
# CompGCN

from pykeen.evaluation import RankBasedEvaluator
from pykeen.hpo import hpo_pipeline

from training_utilities import get_incorrect_tail_predictions, save_to_pkl, read_from_pkl


random_state = 0


'''
指令格式：
    python -u  baselines.py --gpu_id 0  --model  DistMult  --dataset S-DBpedia_small
'''
if '__main__' == __name__:

    parser = argparse.ArgumentParser(description='一个简单的命令行参数示例')
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--training_epochs', default = 100, type = int)
    
    # 解析命令行参数
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f'{pykeen.env()}')

    # metric_keys = ['model_name', 'count', 'variance', 'inverse_median_rank',
    # 'z_geometric_mean_rank', 'arithmetic_mean_rank',
    # 'adjusted_inverse_harmonic_mean_rank', 'harmonic_mean_rank',
    # 'adjusted_arithmetic_mean_rank_index', 'geometric_mean_rank', 'z_arithmetic_mean_rank',
    # 'median_absolute_deviation', 'standard_deviation', 'median_rank', 'z_inverse_harmonic_mean_rank',
    # 'inverse_harmonic_mean_rank', 'adjusted_geometric_mean_rank_index', 'inverse_geometric_mean_rank',
    # 'adjusted_arithmetic_mean_rank', 'inverse_arithmetic_mean_rank', 'hits_at_1', 'hits_at_3', 'hits_at_5',
    # 'hits_at_10', 'z_hits_at_k', 'adjusted_hits_at_k']

    metric_keys = ['model_name', 'count', 'variance', 'inverse_median_rank',
     'arithmetic_mean_rank','adjusted_inverse_harmonic_mean_rank', 'inverse_harmonic_mean_rank',  
     'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10', 'z_hits_at_k', 'adjusted_hits_at_k']

    execution_config = {
        'result_metrics_file': None, #result_metrics.zip
        'metric_keys': metric_keys,
        'model_name': 'Baseline_'+args.dataset+'_'+args.model,
        'model': args.model,
        'training_epochs': args.training_epochs,
        'save_incorrect_classification_triples': False
    }


    # train_filename = Path('training_data/train.txt')

    # train_filename = Path('training_data/BlumBaselineData2/split2_train.txt')
    # val_filename = Path('training_data/valid_split2.txt')
    # test_filename = Path('training_data/test_split2.txt')

    # Path /home/maocy/GRA/datasets/S-DBpedia_GT20E/train.csv
    # ../Datasets/TD1/valid.txt
    # base_Path = '/home/maocy/GRA/datasets/'+args.dataset
    
    base_Path = '../../../datasets/'+args.dataset
    train_filename = Path(base_Path+'/train.txt')
    val_filename = Path(base_Path+'/valid.txt')
    test_filename = Path(base_Path+'/test.txt')

    if args.model == 'CompGCN' :
        create_inverse_triples_FLAG = True
    else:
        create_inverse_triples_FLAG = False

    # creating triples factories for training, testing and validation
    training = TriplesFactory.from_path(
        path = train_filename,
        path_to_numeric_triples = train_filename,
        create_inverse_triples = create_inverse_triples_FLAG
    )

    validation = TriplesFactory.from_path(
        path = val_filename,
        entity_to_id = training.entity_to_id,
        relation_to_id = training.relation_to_id,
        create_inverse_triples = create_inverse_triples_FLAG
    )

    testing = TriplesFactory.from_path(
        path = test_filename,
        entity_to_id = training.entity_to_id,
        relation_to_id = training.relation_to_id,
        create_inverse_triples = create_inverse_triples_FLAG
    )

    if execution_config['result_metrics_file']:
        result_metrics_df = pd.read_csv(execution_config['result_metrics_file'])
    else:
        metric_dict = dict.fromkeys(execution_config['metric_keys'], [])
        result_metrics_df = pd.DataFrame((metric_dict))
    
    # train model using pykeen pipeline
    pipeline_result =  pipeline(
        random_seed = random_state,
        model = execution_config['model'],
        stopper = 'early',
        stopper_kwargs = dict(
            patience = 5
        ),
        evaluator = 'RankBasedEvaluator',
        training = training,
        validation = validation,
        testing = testing,
        training_kwargs= dict(
            num_epochs = execution_config['training_epochs']
        )
    )

    save_path="../../../log"

    # save pipeline results to directory -- can be used to later load trained_model.pkl
    pipeline_result.save_to_directory(f"{save_path}/experiments/pipeline_results/{execution_config['model_name']}")

    model = pipeline_result.model
    # print(pipeline_result.metric_results.to_dict().keys())

    # test set results
    evaluator = RankBasedEvaluator()
    test_results = evaluator.evaluate(
        model = model,
        mapped_triples= testing.mapped_triples,
        # batch_size=1024,
        additional_filter_triples=[
            pipeline_result.training.mapped_triples,
            validation.mapped_triples,
        ],
    )

    # tail prediction results
    test_results_metrics = test_results.to_dict()['tail']['realistic']
    test_results_metrics['model_name'] = execution_config['model_name']
    # result_metrics_df = result_metrics_df.append(test_results_metrics, ignore_index = True)
    result_metrics_df = pd.concat([result_metrics_df, pd.DataFrame([test_results_metrics])], ignore_index=True)
    print(result_metrics_df)

    # save as csv with compression
    compression_opts = dict(method = 'zip', archive_name = f"result_metrics_{execution_config['model_name']}.csv")
    result_metrics_df.to_csv(f"{save_path}/experiments/result_metrics_{execution_config['model_name']}.zip", index = False, compression = compression_opts)

    # store incorrectly predicted triples
    if execution_config['save_incorrect_classification_triples']:
        incorrect_triple_predictions = get_incorrect_tail_predictions(testing, pipeline_result)
        save_to_pkl(data = incorrect_triple_predictions, file_name = f"{save_path}/experiments/incorrect_predictions/{execution_config['model_name']}.pkl")
    










