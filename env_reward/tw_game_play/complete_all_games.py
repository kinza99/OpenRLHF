from api_call import exe_reward_compute
import os
import logging
import numpy as np
from collections import Counter
import re

def calculate_metrics(scores):
    total_scores = len(scores)
    score_counts = Counter(scores)
    score_ratios = {score: count / total_scores for score, count in score_counts.items()}
    if total_scores > 0:
        avg_score = np.mean(scores)
    else:
        avg_score = 0
    print("completion_rewards' Ratios:", score_ratios)
    print("Average completion_reward:", avg_score)

    return score_ratios, avg_score

def complet_each_game(folder_path):
    logging.basicConfig(filename=folder_path+'/'+'game_log_gpt4.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    z8_files = []
    for root, dirs, files in os.walk(folder_path):
        if 'adventure_game.z8' in files:
            z8_files.append(os.path.join(root, 'adventure_game.z8'))

    completion_rewards=[]
    for file_name in z8_files:
        reward=exe_reward_compute(file_name)
        logging.info("="*50)
        logging.info(f"File: {file_name}, Reward: {reward}")
        #logging.info(f"History: {history}")
        completion_rewards.append(reward['completion_reward'])
    logging.info("\n")
    logging.info("-"*50)
    logging.info(calculate_metrics(completion_rewards))

if __name__=="__main__":
    complet_each_game("/cpfs01/shared/llm_code/hesiyang/textworld/data/reward2/game_len11to15")


