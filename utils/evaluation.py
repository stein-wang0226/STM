import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import tqdm

def eval_anomalyDetection(model, data, n_neighbors, batch_size=100):
    m_pred, labels = [], []
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in tqdm.trange(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]
            labels = np.concatenate((labels, labels_batch))

            pos_prob, neg_prob = model.compute_probabilities(sources_batch, destinations_batch, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)
            pred_score = (pos_prob).cpu().numpy()
            m_pred = np.concatenate((m_pred, pred_score.flatten()))
    indices = np.where((labels == 1) | (labels == 0)) #
    labels = labels[indices]
    m_pred = m_pred[indices]
    val_auc = roc_auc_score(labels, m_pred)
    return val_auc