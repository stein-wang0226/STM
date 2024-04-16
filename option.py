import argparse
import sys

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='DGraph', choices=['reddit', 'mooc', 'wikipedia', 'DGraph'])
parser.add_argument('-k', '--k', type=int, help='hop of neighbor', default='1')
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
parser.add_argument('--use_att', action='store_true', help='whether use att layer or not')
parser.add_argument('--node_fetch', action='store_true', help='whether node_fetch or not')
parser.add_argument('--d1', type=int, default=3)
parser.add_argument('--d2', type=int, default=2)
parser.add_argument('--k2', type=int, default=1)
parser.add_argument('--DGraph_size', type=int, default=1100000)
# Spatio-Temporal distance
parser.add_argument('--use_ST_dist', action='store_true', default=True,
                    help='use Spatio-Temporal distance')
parser.add_argument('--type_of_find_k_closest', type=str, default="descending", choices=["ascending", "descending"])


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
