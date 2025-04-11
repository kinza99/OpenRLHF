from openai import OpenAI
import requests
import time
import re
import os 
import torch
import json
# === Configuration ===

TEXTWORLD_SERVER_URL = "http://localhost:8001" 
#MODEL_URL="http://localhost:8000" 
#client = OpenAI(api_key=OPENAI_API_KEY, base_url=f"{MODEL_URL}/v1")
#model="/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/afb2829595f63efa3548e9d6b13aa66e61aa0f38/"
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()
model = "gpt-4o-2024-05-13"

def check_game(game_id):
    response=requests.post(f"{TEXTWORLD_SERVER_URL}/check",params={"game_id": game_id})
    print(f"{game_id}:{response.json()}")
    return response.json()['exist']

def step_game(game_id, command):
    response = requests.post(f"{TEXTWORLD_SERVER_URL}/step", json={
        "game_id": game_id,
        "command": command
    })
    response.raise_for_status()
    return response.json()

def close_game(game_id):
    requests.post(f"{TEXTWORLD_SERVER_URL}/close", params={"game_id": game_id})

def get_status(game_id):
    response = requests.post(f"{TEXTWORLD_SERVER_URL}/get_status", params={
        "game_id": game_id
    })
    response.raise_for_status()
    return response.json()    

def generate_command_with_gpt(history):
    prompt = "\n".join(history + ["Based on the above information, provide the next action command, don't explain, just output the command:"])

    response = client.chat.completions.create(
        model=model,  
        messages=[
            {"role": "system", "content": "You are an AI that excels at playing text adventure games. Only return the command."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def create_games(labels):
    for label in labels:
        print(f"label['game_id']:{label['game_id']}")
        response = requests.post(f"{TEXTWORLD_SERVER_URL}/start", json={
            "game_file": label['GAME_FILE_PATH'],
            "game_id": label['game_id']
        })
        response.raise_for_status()

def create_game(label):
    response = requests.post(f"{TEXTWORLD_SERVER_URL}/start", json={
        "game_file": label['GAME_FILE_PATH'],
        "game_id": label['game_id']
    })
    response.raise_for_status()    


# def environment_func(responses, labels, queries=None):
#     obs=[]
#     for response,label in zip(responses,labels):
#         game_id = label['game_id']
#         if(check_game(game_id))==False:
#             create_game(label)
#             obs.append("")
#         else:
#             result = step_game(game_id, response)
#             obs.append(result['render'] if result['done']==False else None)
#         time.sleep(1)
#     return obs

def environment_func(queries, responses, labels):
    
    obs=[]
    for response,label in zip(responses,labels):
        label_json = json.loads(label)
        game_id = label_json['game_id']
        if(check_game(game_id))==False:
            create_game(label_json)

        result = step_game(game_id, response)
        obs.append(result['observation'].strip() if result['done']==False else None)
    return obs

def collect_responses(labels,obs=None):
    responses=[]
    for label in labels:
        game_id=label['game_id']
        history=get_status(game_id)['history']
        response=generate_command_with_gpt(history)
        responses.append(response)
    return responses #Obs其实用不到 因为game内部维护一个history


def reward_func(queries, responses, labels):
    rewards=[]
    for label in labels:
        label_json = json.loads(label)
        game_id=label_json['game_id']
        completion_reward=get_status(game_id)['score']
        if completion_reward==2:
            final_moves=get_status(game_id)['moves']
            efficiency_reward=label_json['question_length']/final_moves #moves_gt需要load进来
        else:
            efficiency_reward=0

        reward={'completion_reward':completion_reward,'efficiency_rewrd':efficiency_reward}

        rewards.append(completion_reward)
        close_game(game_id)
    return torch.tensor(rewards).to(torch.float32)


def exe(labels,messages):
    alldone=False
    responses=[generate_command_with_gpt(message) for message in messages]

    while not alldone:
        alldone=True
        obs=environment_func(responses,labels)
        responses=collect_responses(labels)
        for obs_single in obs:
            if obs_single!=None:
                alldone=False
                break
        if alldone==True:
            print("ALLDONE!")
            break #需要改为先exe再response

def main(labels,messages):
    exe(labels,messages)
    rewards=reward_func(labels)
    print(rewards)

if __name__=="__main__":
    import json

    file_path = "/cpfs01/shared/llm_code/hesiyang/textworld/data/reward2/textworld.jsonl"

    labels = []
    messages=[]
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            data = json.loads(line)
            data["label"]['game_id']=f"game_{i}"
            labels.append(data["label"])
            messages.append([json.dumps(message_dict) for message_dict in data['message']])


    main(labels,messages)



