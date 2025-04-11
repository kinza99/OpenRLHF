import sqlite3
import time
import torch
import json
from tqdm import tqdm
from multiprocessing import Pool
import ray

@ray.remote
def _execute_with_timeout(db_path, sql, meta_time_out=10.0):
    conn = sqlite3.connect(db_path, timeout=meta_time_out, check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout = {meta_time_out*1000};") 
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall() 
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        cursor.close()  # 关闭游标
        conn.close()  # 关闭连接

# @ray.remote(num_cpus=1)
# def _execute_with_timeout(db_path, sql, meta_time_out=30.0):
#     """使用 func_timeout 执行 SQL 语句"""
#     conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
#     conn.execute(f"PRAGMA busy_timeout = {meta_time_out*1000};") 
#     try:
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         result = cursor.fetchall() 
#     except Exception as e:
#         result = f"Error: {str(e)}"
#     finally:
#         cursor.close()  # 关闭游标
#         conn.close()  # 关闭连接
    
#     return result

def test_execute_with_timeout(db_path, sql):
    """使用 func_timeout 执行 SQL 语句"""
    r = 1
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
    except Exception as e:
        r = 0
    finally:
        cursor.close()
        conn.close()  # 关闭连接
    return r


def execute_sql_time(db_path,sql, meta_time_out=30.0):
    """
    计算执行时间，支持超时
    """
    start_time = time.time()
    try:
        _execute_with_timeout(db_path,sql, meta_time_out)
    except FunctionTimedOut:
        exec_time = float('inf')  # 超时返回无穷大
    except Exception:
        exec_time = float('inf')
    else:
        exec_time = time.time() - start_time
    return exec_time

def iterated_execute_sql( db_path,sql, iterate_num=10, meta_time_out=30.0):
    """
    计算多次执行的平均执行时间，支持超时
    """
    exec_times = []
    for _ in range(iterate_num):
        exec_time = execute_sql_time(db_path,sql, meta_time_out)
        if exec_time != float('inf'):  # 仅统计成功执行的 SQL
            exec_times.append(exec_time)

    avg_time = sum(exec_times) / len(exec_times) if exec_times else float('inf')
    return avg_time

def reward_consist(result_predict,result_gt):
    return 1 if (result_predict==result_gt and 'Error' not in result_predict) else 0

def reward_time(time_predict,time_gt):
    if time_predict=='inf':
        return 0
    return time_gt / time_predict

def reward_final(db_path, ground_truth, predicted_sql, iterate_num, meta_time_out=30.0):
    predicted_result=_execute_with_timeout(db_path,predicted_sql,meta_time_out)
    gt_result=_execute_with_timeout(db_path,ground_truth,meta_time_out)
    reward_first=0
    reward_first=reward_consist(predicted_result,gt_result)
    reward_second=0
    if reward_first:
        print("yessssssss!")
        predicted_time=iterated_execute_sql(db_path,predicted_sql,iterate_num,meta_time_out)
        print(f"predicted_time:{predicted_time}")
        gt_time=iterated_execute_sql(db_path,ground_truth,iterate_num,meta_time_out)
        print(f"gt_time:{gt_time}")
        reward_second=reward_time(predicted_time,gt_time)
    return {'reward_consist':reward_first,'reward_time':reward_second}

def fetch_sql(predicted_results, output_path=None):
    final_sql = {}
    invalid_result = []
    for k, v in predicted_results.items():
        idx = int(k)
        print("------------------- processing {}th example -------------------".format(idx))
        print(v)
        try:
            cot, sql = v.split(': SELECT')
            clean_sql = 'SELECT' + sql
        except Exception as e:
            invalid_result.append(idx)
            clean_sql = 0 # filter resutls without valid SQL, i.e., too long, etc.
        final_sql[k] = clean_sql
    
    if output_path:
        json.dump(final_sql, open(output_path, 'w'), indent=4)
    return final_sql, invalid_result

def post_process(predicted_results):
    split_results = predicted_results.split('is:')
    if len(split_results) > 1:
        clean_sql = split_results[-1]
        clean_sql = clean_sql.strip('\t\n:* ')
        clean_sql = clean_sql.strip('```')
        if clean_sql.startswith('sql'):
            clean_sql = clean_sql[3:]
        clean_sql = clean_sql.strip('\t\n:* ')
    else:
        clean_sql = predicted_results
    return clean_sql

def reward_func(queries, responses, labels):
    rewards = []
    db_paths = []
    pre_sqls = []
    gold_sqls = []
    for query, response, label_str in zip(queries, responses, labels):
        label_json = json.loads(label_str)
        db_path = label_json['db_path']
        label = label_json['ground_truth']
        predicted_sql = post_process(response)
        db_paths.append(db_path)
        pre_sqls.append(predicted_sql)
        gold_sqls.append(label)
    
    try:
        pre_refs = [_execute_with_timeout.remote(db, sql) for db, sql in zip(db_paths, pre_sqls)]
        gold_refs = [_execute_with_timeout.remote(db, sql) for db, sql in zip(db_paths, gold_sqls)]
        ready_pre_refs, pending_pre_refs = ray.wait(pre_refs, num_returns=len(pre_refs), timeout=60.0)
        ready_gold_refs, pending_gold_refs = ray.wait(gold_refs, num_returns=len(gold_refs), timeout=60.0)
        for i in range(len(pre_refs)):
            if pre_refs[i] in ready_pre_refs and gold_refs[i] in ready_gold_refs:
                pre_result = ray.get(pre_refs[i])
                gold_result = ray.get(gold_refs[i])
                rewards.append(reward_consist(pre_result, gold_result))
            else:
                rewards.append(0)
    finally:
        for ref in pre_refs:
            ray.cancel(ref, force=True)
        for ref in gold_refs:
            ray.cancel(ref, force=True)
    return torch.tensor(rewards).to(torch.float32)

def environment_func(queries, responses, labels):
    obs = []
    db_paths = []
    pre_sqls = []
    start_time = time.time()
    for query, response, label_str in zip(queries, responses, labels):
        label_json = json.loads(label_str)
        db_path = label_json['db_path']
        predicted_sql = post_process(response)
        db_paths.append(db_path)
        pre_sqls.append(predicted_sql)
    try:
        pre_refs = [_execute_with_timeout.remote(db, sql) for db, sql in zip(db_paths, pre_sqls)]
        ready_refs, pending_refs = ray.wait(pre_refs, num_returns=len(pre_refs), timeout=60.0)
        pre_results = []
        for r in pre_refs:
            if r in ready_refs:
                pre_results.append(ray.get(r))
            else:
                pre_results.append("Error: Query execution timed out after seconds")
        for pre_result in pre_results:
            if isinstance(pre_result, str) and 'Error' in pre_result:
                obs.append(pre_result)
            else:
                obs.append(None)
    finally:
        for ref in pre_refs:
            ray.cancel(ref, force=True)
    end_time = time.time()
    print(f"环境函数执行时间: {end_time - start_time:.4f} 秒")
    return obs

    
def process_item(item):
    db_path = '/cpfs01/shared/llm_code/hesiyang/data/bird/train/train_databases/' + item['db_id'] + '/' + item['db_id'] + '.sqlite'
    return _execute_with_timeout(db_path, item['SQL'])

def main():
    with open('/cpfs01/shared/llm_code/hesiyang/data/bird/train/train.json', 'r') as f:
        data = json.load(f)
    
    
    # 使用可用CPU核心数量，也可以手动设置，例如 n_processes = 4
    n_processes = 16
    
    print(f"启动 {n_processes} 个进程进行处理...")
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(process_item, data), total=len(data)))
    



if __name__ == '__main__':
    main()
    # main_chunk()
#     import random
#     import json
    # db_path = '/cpfs01/shared/llm_code/hesiyang/data/bird/train/train_databases/talkingdata/talkingdata.sqlite'
    # print(generate_schema_prompt(db_path))
    # # sql = "SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID WHERE T1.OrderDate LIKE '1997%'"
    # sql = "SELECT app_id FROM app_events WHERE is_active = 1 AND is_installed = 1"
    # print(sql)
    # # sql = "SELECT name FROM sqlite_master WHERE type='table'"
    # result = _execute_with_timeout(db_path, sql, meta_time_out=30.0)
    # print(isinstance(result, str))

    # with open('/cpfs01/shared/llm_code/hesiyang/data/bird/train/train.json', 'r') as f:
    #     data = json.load(f)
    # right = 0
    # wrong = 0
    # for i in tqdm(range(len(data))):
    #     db_path = '/cpfs01/shared/llm_code/hesiyang/data/bird/train/train_databases/' + data[i]['db_id'] + '/' + data[i]['db_id'] + '.sqlite'
    #     if not test_execute_with_timeout(db_path, data[i]['SQL'], meta_time_out=10.0):
    #         wrong += 1
    #     else:
    #         right += 1
    # print(f"right: {right}, wrong: {wrong}")
