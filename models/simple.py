
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
from models import *
# from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataset import *
from tqdm import tqdm
from math import sqrt, log
from executors import HumanevalExecutor, AppsExecutor
from ChatModels import GPT35Chat

class Simple:
    def __init__(self, args):
        self.args = args
        self.sample_times = 0
        self.gamma = 0.9
        # if 'gpt3.5' in args.arch or 'gpt4' in args.arch:
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.generator = GPT35Chat(args.arch, self.tokenizer, args)

        if args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        elif args.dataset == 'apps':
            self.executor = AppsExecutor(args)
        elif args.dataset == 'codeforces':
            self.executor = AppsExecutor(args)

    def generate(self, problem_instance):
        st = time.time()
        self.cur_prob_instance = problem_instance
        
        raw_prompt = problem_instance['prompt']
        # raw_prompt += "\nAssume you are proficient in solving Python programming problems. For the above problem, please analyze and think step by step, and then provide the final code that solves the problem."


        with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
            f"Generate the code ONLY. No other explanation or words attached!\n") + raw_prompt
        # print(with_instru_input_prompt)
        # assert 0
        complete_program = self.generator.generate_response_api(
            prompt=with_instru_input_prompt,
            top_k=1,
            max_length=self.args.horizon,
            temperature=0.0
        )[0]

        train_rewards = [0.0]
        test_rewards = [self.get_reward(complete_program)]

        output_dict = {
            'final_program': complete_program,
            'train_reward': train_rewards[0],
            'test_reward': test_rewards[0],
            'all_programs': [complete_program],
            'all_train_rewards': train_rewards,
            'all_test_rewards': test_rewards,
            'avg_sample_time': time.time() - st,
            'debug_log': []
        }

        return output_dict

    def get_reward(self, program):
        # 计算pass rate
        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, program, 'test')
            
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            curr_res = []
        
        assert isinstance(curr_res, list)
        pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
        reward = pass_rate
        return reward
