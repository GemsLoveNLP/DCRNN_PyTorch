from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import torch

def main(args):

    print("-"*25)

    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        if args.resume is not None:
            # supervisor._epoch_num = 92
            supervisor.load_model2()

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--resume', default=None, type=str, help='Old .tar Checkpoint file to continue from')
    args = parser.parse_args()
    main(args)
