# -*- coding:utf-8 _*-
# Process directory
import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')
import shutil
import json
# CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'  # 必须放在import各种python的包之前运行

import openai
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] ="spawn"

API_KEY = "sk-xxxx"

openai.api_key = API_KEY

os.environ["http_proxy"] = "http://127.0.0.1:8888"
os.environ["https_proxy"] = "http://127.0.0.1:8888"
os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
os.environ["OPENAI_API_KEY"] = API_KEY

# imports
import torch
import argparse
# from Processor import Processor
from argparse import ArgumentParser
from accelerate import Accelerator
from utils import *
import pprint
import wandb
from models import *
import time
from ChatModels import *
from dataset import *
from refine import *
from executors import *
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def parse_args():
    parser = ArgumentParser("BER")
    parser.add_argument('-d', '--dataset', type=str, default='humaneval', choices=['humaneval', 'apps', 'codeforces'])
    parser.add_argument('-m', '--model', type=str, default='simple', choices=['simple',  'cothone', 'pseudohone', 'keypointshone'])
    parser.add_argument("--save", type=str, default="./results2", help="Directory to save generated code.")
    parser.add_argument('-eid', '--experiment-idx', type=int, default=57, help='Experiment id for one model')
    parser.add_argument("--arch", default="gpt3.5", choices=["gpt2", "gpt-neo", "gpt3.5", "gpt3.5completion", 'kimi', 'gpt4','gpt4o-mini','deepseek','qwen','qwencoder', 'qwen14b', 'claude'])
    parser.add_argument("--loadArchDir", default=f"", type=str)
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cudaDevice', type=str, default='cuda')

    parser.add_argument('--rerun', action='store_true', default=False, help="If True, rerun if the output file already exists.")
    parser.add_argument("-i", "--index", default=None, type=int)  # 81 is hard, could be used for single test
    parser.add_argument("--start", default=4000, type=int)
    parser.add_argument("--end", default=4100, type=int)
    parser.add_argument("--ts-mode", default="best", choices=["best", "sample"], help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")
    parser.add_argument("--horizon", default=4096, type=int, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search or PG-TD.")
    parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--width", default=3, type=int, help="The maximum number of children for any node.")
    parser.add_argument('--top-k-cache-steps', type=int, default=1024, help="Number of forward steps to cache top k caches, default 1024 means the whole horizon.")
    parser.add_argument("--public-cases-type", type=str, default='half', help="Number of public test cases to use for evaluation.")
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--max_iters', type=int, default= 5 )
    parser.add_argument('--total_input_token_num', type=int, default=0, help='The maximum number of tokens to input.')
    parser.add_argument('--total_output_token_num', type=int, default=0, help='The maximum number of tokens to output.')
    parser.add_argument('--failed_json_num', type=int, default=0, help='The number of failed json format output.')
    parser.add_argument('--all_json_num', type=int, default=0, help='The number of all json format output.')
    parser.add_argument('--verbal_length_exd_num', type=int, default=0, help='The number of length of verbal length too long')
    parser.add_argument('--verbal_length_check_num', type=int, default=0, help='The number of length check of verbal feedback')
    parser.add_argument('--rollout_count', type=int, default=-1, help='The rollout count')
    parser.add_argument('--generate_tests_total', type=int, default=0, help='The generate tests count')
    parser.add_argument('--failed_generate_tests_count', type=int, default=0, help='The failed generated tests count total')
    
    parser.add_argument('--aug', type=str, default='none', choices=['demonstration', 'retrieval', 'none'])
    parser.add_argument("--rollout", default=1, type=int, help="The maximum number of rollouts for PG-TD.")
    # parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search or PG-TD.")
    # parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--max-sample-times", default=768, type=int, help="The maximum number of Transformer generation function calls.")
    parser.add_argument("--appsdifficulty", default="introductory",type = str, choices=['introductory', 'interview', 'competition','800', '900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700','1800', '1900', '2000']) 
    parser.add_argument("--scale", default="full",type = str, choices=['full', 'partial']) 



    # parser.add_argument('--explanation', action='store_true', default=False, help="sketch debug add explanation")
    parser.add_argument('--sample_type', default="none", type=str)
    parser.add_argument('--local_callmode', default="load", type=str, choices=['load', 'api'])
    parser.add_argument('--apiport',type=int, default=8000)

    # New GPU argument
    parser.add_argument('--gpu', type=str, default='0', help="Specify GPU ID (default: 0)")


    args = parser.parse_args()
    return args


def kwargs_wrapper_gen(func, delete_keys=[], add_keys={}):
    def kwargs_wrapper(**kwargs):
        for key in delete_keys:
            del kwargs[key]
        for key in add_keys:
            kwargs[key] = add_keys[key]
        return func(**kwargs)
    return kwargs_wrapper

def strategy_factory(strategy: str):
    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple)
    elif strategy == 'cothone':
        return kwargs_wrapper_gen(run_cothone)
    elif strategy == 'pseudohone':
        return kwargs_wrapper_gen(run_pseudohone)
    elif strategy == 'keypointshone':
        return kwargs_wrapper_gen(run_keypointshone)
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")

import logging

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "experiment.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.loadArchDir = f"{args.arch}"
    if args.experiment_idx == 2:
        args.rerun = True

    args.save = f"{args.save}/{args.dataset}/Experiment_{args.experiment_idx}"
    print(f'save dir: {args.save}')

    log_dir = os.path.join(args.save, "logs")
    setup_logging(log_dir)

    if args.rerun:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
            print(f"'{args.save}' has been removed and recreated.")
    
    os.makedirs(args.save, exist_ok=True)
    
    run_strategy = strategy_factory(args.model)
    run_strategy(**vars(args))


    
    
def run_simple(**kwargs):
    args = argparse.Namespace(**kwargs)
    st = time.time()
    model = Simple(args)
    problem_indices, data_handler = data_loader(args)
    go_generation(args, model, problem_indices, data_handler)



def run_cothone(**kwargs):
    args = argparse.Namespace(**kwargs)
    st = time.time()
    model = CoTHoner(args)
    problem_indices, data_handler = data_loader(args)
    go_generation(args, model, problem_indices, data_handler)


    
def run_pseudohone(**kwargs):
    args = argparse.Namespace(**kwargs)
    st = time.time()
    model = PseudoHoner(args)
    problem_indices, data_handler = data_loader(args)
    go_generation(args, model, problem_indices, data_handler)
    
def run_keypointshone(**kwargs):
    args = argparse.Namespace(**kwargs)
    st = time.time()
    model = KeypointsHoner(args)
    problem_indices, data_handler = data_loader(args)
    go_generation(args, model, problem_indices, data_handler)
    

    
def data_loader(args):
    problem_indices = []
    removed_list = []
    problem_indices = range(args.start, args.end)
    if args.dataset == 'apps':
        if args.appsdifficulty == 'introductory':
            # args.start = 4000
            # args.end = 4100
            problem_indices = [i for i in range(args.start, args.end)]
            if args.scale == 'partial':
                problem_indices = [4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099]
        elif args.appsdifficulty == 'interview':
            # args.start = 0
            # args.end = 100
            problem_indices = [i for i in range(args.start, args.end)]

        elif args.appsdifficulty == 'competition':
            # args.start = 3000
            # args.end = 3100
            problem_indices = [i for i in range(args.start, args.end)]
        data_handler = APPSHandler(problem_indices, args)
    elif args.dataset == 'humaneval':
        if args.index is None:
            problem_indices = [i for i in range(0,164)]
            # problem_indices = [163]
        data_handler = HumanevalHandler(problem_indices, args)
    elif args.dataset == 'codeforces':
        data_handler = CodeforcesHandler(problem_indices, args)
    if args.dataset == 'codeforces':
        problem_indices = [problem_instance['index'] for problem_instance in data_handler.problems]

    return problem_indices, data_handler

import time
import os
import json
import logging
import concurrent.futures

def generate_with_timeout(model, problem_instance,idx,  timeout=1800):

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.generate, problem_instance)
        try:
            output_dict = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Generation timed out: problem_{idx}.")
            return None
        except Exception as e:
            print(f"Error occurred while generating problem_{idx}: {str(e)}")
            return None
    return output_dict

def go_generation(args, model, problem_indices, data_handler):
    for idx, problem_instance in zip(problem_indices, data_handler.problems):
        st = time.time()
        result_loc = os.path.join(args.save, f"{idx}.json")
        debug_loc = os.path.join(args.save, f"{idx}_debug.json")
        if not args.rerun and os.path.exists(result_loc):
            print(f"Found {result_loc}, args.rerun not enabled, skipping")
            continue
        print(f"Solving Problem #{idx}")
        

        output_dict = generate_with_timeout(model, problem_instance, idx)
        
        if output_dict is None:

            continue

        # Log prompt and generated code
        logging.info(f"Problem #{idx} Prompt: {problem_instance['prompt']}")
        logging.info(f"Problem #{idx} Generated Code: {output_dict['final_program']}")

        print(f"Final Program: \n{output_dict['final_program']}")
        print(f"train rewards: {output_dict['train_reward']}")
        print(f'test rewards: {output_dict["test_reward"]}')
        print(f'solve time: {time.time() - st}')
        
        with open(result_loc, "w") as f:
            json.dump({
                'codes': output_dict['final_program'],
                'rewards': output_dict['test_reward'],
                'train rewards': output_dict['train_reward'],
                'time': time.time() - st,
                'all_train_rewards': output_dict['all_train_rewards'],
                'all_test_rewards': output_dict['all_test_rewards'],
                'input_token_num': args.total_input_token_num,
                'output_token_num': args.total_output_token_num,
            }, f)


        with open(debug_loc, "w") as f:
            json.dump(output_dict['debug_log'], f, indent=4)

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
