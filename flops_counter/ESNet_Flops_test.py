import sys
import argparse

from model.ESNet import ESNet

import torch

from flops_counter.ptflops import get_model_complexity_info

pt_models = {
    'ESNet': ESNet,

    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='ESNet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        net = pt_models[args.model](num_classes=20).cuda()

        flops, params = get_model_complexity_info(net, (3, 512, 1024),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        print('Flops: ' + flops)
        print('Params: ' + params)
