import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

from utils.evaluation import eval_anomalyDetection
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data_Dgraphfin
import tqdm
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables

parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='DGraph', choices=['reddit', 'mooc', 'wikipedia', 'DGraph'])
parser.add_argument('-k','--k', type=int, help='hop of neighbor', default='1')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--path_prefix', type=str, default='./utils/data/')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=5, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--embedding_module', type=str, default="HopTransformer", choices=[
    "graph_attention", "graph_sum", "identity", "time", "HopTransformer"], help='Type of embedding module')

parser.add_argument('--agg_time_line', type=str, default="auto-encoder", choices=['linear', 'auto-encoder'])

parser.add_argument('--use_time_line', action='store_true')
parser.add_argument('--time_line_length', type=int, default=1)
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--hard_sample', action='store_true')
parser.add_argument('--use_att',  action='store_true', help='whether use att layer or not')
parser.add_argument('--node_fetch',  action='store_true', help='whether node_fetch or not')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

Path(args.path_prefix + "saved_models/").mkdir(parents=True, exist_ok=True)
Path(args.path_prefix + "saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'{args.path_prefix}saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
        epoch: f'{args.path_prefix}saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
log_filename = 'log/{}-{}.log'.format(
    args.prefix,
    '--'.join([args.data, args.embedding_module, f'k={args.k}', f'time_line_length={args.time_line_length}',f'node_fetch={args.node_fetch}',f'use_att={args.use_att}', f'n_degree={args.n_degree}',f'bs={args.bs}',f'ls={args.lr}', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')])
)
fh = logging.FileHandler(log_filename)

fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

print(args.data)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, test_data, val_data = \
    get_data(dataset_name=args.data,node_fetch= args.node_fetch)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data,
                                       sample_mode=args.sample_mode,
                                       hard_sample=args.hard_sample)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data,
                                      sample_mode=args.sample_mode,
                                      hard_sample=args.hard_sample)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# logger.info("第{}个k".format(args.k))
for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    #
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT,
              embedding_module_type=args.embedding_module,
              n_neighbors=NUM_NEIGHBORS,
              use_time_line=args.use_time_line,
              time_line_length=args.time_line_length, agg_time_line=args.agg_time_line, k=args.k,
              data=args.data,
              use_att=args.use_att)  # todo agg time

    criterion = torch.nn.BCELoss()
    if DATA == "DGraph":
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)

    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []
    max_val_auc = 0
    max_test_auc = 0
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training
        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm.trange(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()
            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                # todo 这个才是异常检测的
                labels_batch = train_data.labels[start_idx:end_idx]
                size = len(sources_batch)
                tgn = tgn.train()
                # 这里的negatives_batchs有什么用，因为这是对未来的边预测，不是异常检测，所以要设置这个
                pos_prob, neg_prob = tgn.compute_probabilities(sources_batch, destinations_batch,
                                                               timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

                # todo 这个是边预测的，不是异常检测
                # loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
                # todo 这个是异常检测的
                if args.data == "DGraph":
                    # labels_batch = labels_batch.to(device)
                    labels_batch = torch.from_numpy(labels_batch.astype(np.float32)).to(device)
                else:
                    labels_batch = torch.from_numpy(labels_batch.astype(np.float32)).to(device)
                loss += criterion(pos_prob.squeeze(), labels_batch)

            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### todo Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        val_auc = eval_anomalyDetection(model=tgn,
                                        data=val_data,
                                        n_neighbors=NUM_NEIGHBORS,
                                        batch_size=args.bs)

        test_auc = eval_anomalyDetection(model=tgn,
                                         data=test_data,
                                         n_neighbors=NUM_NEIGHBORS,
                                         batch_size=args.bs)
        max_val_auc = max(max_val_auc, val_auc)
        max_test_auc = max(max_test_auc,test_auc)
        logger.info("val auc:{}".format(val_auc))
        logger.info(
            'test auc: {}'.format(test_auc))
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))

        torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # ### todo Test
    # tgn.embedding_module.neighbor_finder = full_ngh_finder
    # test_auc = eval_anomalyDetection(model=tgn,
    #                                  data=test_data,
    #                                  n_neighbors=NUM_NEIGHBORS,
    #                                  batch_size=args.bs)
    logger.info(
        'max test auc: {}'.format(max_test_auc))
    logger.info(
        'max val auc: {}'.format(max_val_auc))
    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TGN model')
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')