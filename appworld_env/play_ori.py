import requests
import json
import os
from openai import OpenAI
import torch


SERVER_URL = "http://127.0.0.1:8003" 
max_interactions=50

def environment_func(queries, responses, labels):
    breakpoint()
    """
    向 game_server 发送请求，获取下一步的响应。
    """
    labels_json = []
    for label in labels:
        tmp = json.loads(label)
        labels_json.append({"experiment_name": tmp["experiment_name"], "task_id": tmp["task_id"]})
    url = f"{SERVER_URL}/step"
    payload = {
        "responses": responses,
        "labels": labels_json
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 如果响应状态码不是 200，会抛出异常
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def reward_func(queries, responses, labels):
    """
    向 game_server 发送请求，获取奖励（reward）。
    """
    url = f"{SERVER_URL}/get_reward"
    payload = {
        "labels": labels
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        return torch.tensor(response.json()["rewards"])
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

client = OpenAI()
model = "gpt-4o-2024-05-13"
history_dict = {}

def call_llm(messages: list[dict]) -> str:
    """
    Call an LLM with a history of messages and return the response.
    """
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=400, seed=123
    )
    text_response = ""
    if response.choices:
        text_response = response.choices[0].message.content
    return text_response

def collect_responses(labels, obs):
    responses = []

    for label, single_obs in zip(labels, obs):
        experiment_name = label['experiment_name']
        task_id = label['task_id']
        
        history = history_dict[(experiment_name, task_id)]
        
        if single_obs:
            history.append({"role": "assistant", "content": single_obs, "type": "text"})

        for message in history:
            if "type" not in message:
                print("add")
                message["type"] = "text" 
            # 如果 content 是一个列表，确保每个元素都包含 "type"
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if "type" not in item:
                        item["type"] = "text"  
                message["content"] = " ".join([str(item) for item in message["content"]])  
        
        new_response = call_llm(history)  # 使用历史记录生成新的响应
        history.append({"role": "assistant", "content": new_response, "type": "text"})

        history_dict[(experiment_name, task_id)] = history
        responses.append(new_response)  # 将新的响应加入到 responses 列表中

    return responses

def exe(labels,messages):
    for label, message in zip(labels, messages):
        experiment_name = label['experiment_name']
        task_id = label['task_id']
        
        # 初始化每个 experiment_name 和 task_id 的 history
        if (experiment_name, task_id) not in history_dict:
            history_dict[(experiment_name, task_id)] = [{"role": "user", "content": message, "type": "text"}]
    
    obs = [""] * len(labels)

    i=0

    for _ in range(max_interactions):
        print(f"第{i}次迭代！")
        i=i+1
        alldone=True
        responses=collect_responses(labels,obs) #这里会把obs加到history上 然后查llm
        print(responses)
        obs=environment_func(None,responses,labels)
        print("-"*50)
        print(f"obs:{obs}")
        print(F"environment_func ended!")
        for single_obs in obs:
          if single_obs!=None:
            # print(f"None!")
            alldone=False
            break
        if alldone==True: 
          break

def main(labels,messages):
  exe(labels,messages)
  rewards=reward_func(None, None, labels)
  print(f"rewards:{rewards}")

import json

def read_task_ids_from_jsonl(file_path, num_lines=2):
    task_ids = []
    messages=[]
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= num_lines:
                break
            data = json.loads(line) 
            label = json.loads(data['label'])
            task_ids.append(label['task_id'])
            messages.append(data['message']) 
    return task_ids,messages

if __name__ == "__main__":
    experiment_name = "minimal_react_agent_test_normal_gpt_trysplitexereward"

    file_path = "/cpfs01/user/duhe/OpenRLHF/data/appworld_train.jsonl"
    task_ids,messages = read_task_ids_from_jsonl(file_path)
    
    labels = [json.dumps({'experiment_name': experiment_name, 'task_id': task_id}) for task_id in task_ids]
    
    # Call main with the labels
    main(labels,messages)  