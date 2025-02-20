import os
import json
from utils import *

def reset_test_cases(problem_instance):
    train_in_outs, test_in_outs = {}, {}

    total_in_outs = {
        'inputs': [],
        'outputs': []
    }
    for i, input_data in enumerate(problem_instance["train_in_outs"]['inputs']):
        total_in_outs['inputs'].append(input_data)
        total_in_outs['outputs'].append(problem_instance["train_in_outs"]['outputs'][i])
    for i, input_data in enumerate(problem_instance["test_in_outs"]['inputs']):
        total_in_outs['inputs'].append(input_data)
        total_in_outs['outputs'].append(problem_instance["test_in_outs"]['outputs'][i])

    public_test_cases_num = min(15, len(total_in_outs['inputs']) // 2)
    private_test_cases = len(total_in_outs['inputs']) - public_test_cases_num
    if public_test_cases_num < 1 or private_test_cases < 1:
        raise Exception(f"Not enough test cases: {public_test_cases_num}, {private_test_cases}.")
    train_in_outs['inputs'] = total_in_outs['inputs'][:public_test_cases_num]
    train_in_outs['outputs'] = total_in_outs['outputs'][:public_test_cases_num]
    test_in_outs['inputs'] = total_in_outs['inputs'][public_test_cases_num:]
    test_in_outs['outputs'] = total_in_outs['outputs'][public_test_cases_num:]
    return train_in_outs, test_in_outs


class CodeforcesHandler:
    def __init__(self, problem_indices, args):
        self.problems = []
        base_dir = os.path.join(get_proj_path(), "dataProcess", "codeforces")

        # 如果指定了 appsdifficulty，则只加载对应 rating 文件夹
        if args.appsdifficulty is not None:
            target_dir = os.path.join(base_dir, str(args.appsdifficulty))
            if not os.path.exists(target_dir):
                raise FileNotFoundError(f"Rating folder {target_dir} does not exist.")

            # 处理指定难度文件夹内的所有题目文件
            self._load_problems_from_dir(target_dir, problem_indices)
        else:
            # 如果未指定 difficulty，则加载所有题目
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    self._load_problems_from_dir(os.path.join(root, dir_name), problem_indices)

    def _load_problems_from_dir(self, directory, problem_indices):
        problem_count = 0

        # 按文件名中的数字顺序排序
        files = [file for file in os.listdir(directory) if file.endswith(".json")]
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))  # 按数字排序

        for file in files:
            json_path = os.path.join(directory, file)
            try:
                # 打开并读取 JSON 文件
                with open(json_path, 'r', encoding='utf-8') as f:
                    problem_instance = json.load(f)
            except Exception as e:
                print(f"Failed to read {json_path}: {e}")
                continue

            problem_instance['code_type'] = 'standard_input'
            problem_instance['method_name'] = None

            if problem_count > max(problem_indices):
                continue

            problem_instance['index'] = problem_count
            problem_instance['task_id'] = f"APPS/{problem_count}"
            # problem instance formation
            input_prompt = "\nQUESTION:\n"
            input_prompt += problem_instance['full_description']
            input_prompt += "\nUse Standard Input format"  # \n"
            input_prompt += "\nANSWER:\n"
            problem_instance["prompt"] = input_prompt

            # test cases for train and test
            try:
                train_in_outs, test_in_outs = reset_test_cases(problem_instance)
            except Exception as e:
                print(f"Skipping problem {json_path} due to insufficient test cases: {e}")
                continue

            problem_instance["train_in_outs"] = train_in_outs
            problem_instance["test_in_outs"] = test_in_outs
            if problem_count >= min(problem_indices):
                self.problems.append(problem_instance)
            problem_count += 1
