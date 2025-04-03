from reward import Reward
from env import ENV
from generate import collect_oneresponse_from_gpt
import argparse
import json
import fnmatch
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import os
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import math
import time
from func_timeout import func_timeout, FunctionTimedOut
import backoff
import openai
import pandas as pd
import sqlparse
from tqdm import tqdm

def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] +'.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list


def package_sqls(sql_path, db_root_path, mode='gt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '.sql') #这里之前是_gold.sql 有问题
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

from env_reward import reward_final
def evaluate_sql_response(db_path, question, ground_truth, api_key, engine, idx, iterate_num,meta_time_out, knowledge=None):
    """
    针对单个question生成sql并计算reward
    """
    # 生成 SQL
    predicted_sql = collect_oneresponse_from_gpt(db_path, question, api_key, engine, knowledge)
    # 计算 reward
    # env=ENV(db_path)
    # predicted_result=env.execute_sql(predicted_sql,meta_time_out)
    # gt_result=env.execute_sql(ground_truth,meta_time_out)
    # reward=Reward()
    # reward_first=0
    # reward_first=reward.reward_consist(predicted_result,gt_result)
    # reward_second=0
    # if reward_first:
    #     print("yessssssss!")
    #     predicted_time=env.iterated_execute_sql(predicted_sql,iterate_num,meta_time_out)
    #     print(f"predicted_time:{predicted_time}")
    #     gt_time=env.iterated_execute_sql(ground_truth,iterate_num,meta_time_out)
    #     print(f"gt_time:{gt_time}")
    #     reward_second=reward.reward_time(predicted_time,gt_time)
    # return {'reward_consist':reward_first,'reward_time':reward_second}
    return reward_final(db_path, ground_truth,predicted_sql, iterate_num,meta_time_out)

def evaluate_sql_responses(db_path_list, question_list,ground_truth_list, api_key, engine, iterate_num,meta_time_out,knowledge_list=None):
    rewards=[]
    for i, question in tqdm(enumerate(question_list)):
        reward=evaluate_sql_response(db_path_list[i],question,ground_truth_list[i],api_key, engine, i,iterate_num,meta_time_out,knowledge=None)
        print(f"*********reward:{reward}*****************")
        rewards.append(reward)
    return rewards


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--test_path', type=str, default='')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')

    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--result_output_file',type=str,required=True)
    args_parser.add_argument('--iterate_num',type=str,default=10)
    args = args_parser.parse_args()
    
    eval_data = json.load(open(args.eval_path, 'r'))
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)
    print(f"questionlist长度：{len(question_list)}")
    ground_truth_list, _ = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)
    assert len(question_list) ==len(ground_truth_list)
    rewards=evaluate_sql_responses(db_path_list, question_list,ground_truth_list, args.api_key, args.engine,args.iterate_num, args.meta_time_out,knowledge_list=None)

    file_path = args.result_output_file

    # 将 rewards 字典写入 JSON 文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rewards, f, indent=4, ensure_ascii=False)

    print(f"rewards 数据已成功写入 {file_path}")
    print('successfully collect results from {} for {} evaluation; Use knowledge: {}; Use COT: {}'.format(args.engine, args.mode, args.use_knowledge, args.chain_of_thought))
