import os
import json

def calculate_metrics(directory, start, end):
    total_files = 0
    correct_files = 0
    total_rewards = 0.0

    for filename in os.listdir(directory):
        if filename.endswith(".json") and '_debug' not in filename:
            try:
                problem_number = int(filename.split('.')[0])
                if start <= problem_number <= end:
                    total_files += 1
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    

                    train_rewards = data.get('all_train_rewards', [])
                    test_rewards = data.get('all_test_rewards', [])
                    
                    if len(train_rewards) == len(test_rewards) and len(train_rewards) > 0:

                        max_train_reward = max(train_rewards)
                        max_train_reward_index = len(train_rewards) - 1 - train_rewards[::-1].index(max_train_reward)
                        

                        reward = test_rewards[max_train_reward_index]
                        total_rewards += reward


                        if reward == 1.0:
                            correct_files += 1

            except (ValueError, IndexError):
                continue

    if total_files == 0:
        return 0.0, 0.0

    pass_at_1 = correct_files / total_files
    pass_rate = total_rewards / total_files
    return pass_at_1, pass_rate, total_files

if __name__ == "__main__":
    directory_path = "results2/apps/Experiment_1"  

    start_problem = 0  
    end_problem = 4099 

    pass_at_1, pass_rate, total_files = calculate_metrics(directory_path, start_problem, end_problem)
    print(f"Prob {start_problem} to {end_problem}")
    print(f"Total Nums: {total_files}")
    print(f"Pass@1: {pass_at_1 * 100:.2f}%")
    print(f"Pass Rate: {pass_rate * 100:.2f}%")