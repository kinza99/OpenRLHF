from openai import OpenAI
import requests
import time
import re

# === Configuration ===
OPENAI_API_KEY = ""
TEXTWORLD_SERVER_URL = "http://localhost:8001" 
MODEL_URL="http://localhost:8000" 
client = OpenAI(api_key=OPENAI_API_KEY, base_url=f"{MODEL_URL}/v1")
model="/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/afb2829595f63efa3548e9d6b13aa66e61aa0f38/"



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


def environment_func(responses, labels, queries=None):
    obs=[]
    for response,label in zip(responses,labels):
        game_id = label['game_id']
        result = step_game(game_id, response)
        obs.append({'output':result['render'],'iscompleted':result['done']})
        time.sleep(1)
    return obs

def collect_responses(labels,obs=None):
    responses=[]
    for label in labels:
        game_id=label['game_id']
        history=get_status(game_id)['history']
        # response=generate_command_with_gpt(history)
        response=input("请输入命令:")
        responses.append(response)
    return responses #Obs其实用不到 因为gamen内部维护一个history


def reward_func(labels, queries=None, responses=None):
    rewards=[]
    for label in labels:
        game_id=label['game_id']
        completion_reward=get_status(game_id)['score']
        if completion_reward==2:
            final_moves=get_status(game_id)['moves']
            efficiency_reward=label['quest-length']/final_moves #moves_gt需要load进来
        else:
            efficiency_reward=0

        reward={'completion_reward':completion_reward,'efficiency_rewrd':efficiency_reward}

        rewards.append(reward)
        close_game(game_id)
    return rewards


def exe(labels):
    create_games(labels)
    alldone=False
    while not alldone:
        alldone=True
        responses=collect_responses(labels)
        obs=environment_func(responses,labels)
        for obs_single in obs:
            if obs_single['iscompleted']==False:
                alldone=False
                break
        if alldone==True:
            break

def main(labels):
    exe(labels)
    rewards=reward_func(labels)
    print(rewards)

if __name__=="__main__":
    import json

    file_path = "/cpfs01/shared/llm_code/hesiyang/textworld/data/reward2/textworld.jsonl"

    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1:
                break
            data = json.loads(line)
            data["label"]['game_id']=f"game_{i}"
            labels.append(data["label"])

    main(labels)



