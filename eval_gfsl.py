import numpy as np
import torch
from model.evaluator.gfsl_evaluator import GFSLEvaluator
from model.utils import (
    pprint, set_gpu,
    get_eval_command_line_parser,
    postprocess_eval_args,
)
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_eval_command_line_parser()
    args = postprocess_eval_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    set_gpu(args.gpu)
    evaluator = GFSLEvaluator(args)
    if args.criteria == 'AUSUC':
        evaluator.best_bias_acc = 0
        evaluator.best_bias_map = 0
    else:
        evaluator.get_calibration()
    evaluator.evaluate_gfsl()
    evaluator.final_record()


