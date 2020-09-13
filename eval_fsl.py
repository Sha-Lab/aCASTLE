import numpy as np
import torch
from model.evaluator.fsl_evaluator import FSLEvaluator
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
    evaluator = FSLEvaluator(args)
    evaluator.evaluate_fsl()
    evaluator.final_record()
    # print(args.save_path)



