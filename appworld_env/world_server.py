import logging
from openai import OpenAI
import re
from jinja2 import Template
from appworld.task import Task
from appworld.serve.environment import run as environment_serve
from appworld import evaluate_task
from appworld import AppWorld
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import socket


app = FastAPI()

worlds={} #worlds是字典类型所以可以直接通过实验名称和对应的taskid找到对应的world
ports={}

stack = list(range(20000,30000))
def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    """
    检查指定端口是否被占用
    
    Args:
        port (int): 要检查的端口号
        host (str): 要检查的主机地址，默认为本地回环地址
    
    Returns:
        bool: 如果端口被占用返回 True，否则返回 False
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True

def get_next_port():
  while True:
    if stack:
      next_port=stack.pop()
      if not is_port_in_use(next_port):
        return next_port
    else:
        raise Exception("No more ports available.")

import time
def start_server(port):
    if not isinstance(port, list):
      # 调用外部命令静默启动服务器
      command = f"appworld serve environment --port {port}"
      subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
      for p in port:
        command = f"appworld serve environment --port {p}"
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(5)
      # environment_serve(port=port)
      print(f"{port}server on!")


def environment_func_single(response,label,query=None):
  
  if label['experiment_name'] not in worlds or label['task_id'] not in worlds[label['experiment_name']]:
    # port = get_next_port()
    # start_server(port)
    port=ports[label['experiment_name']][label['task_id']]
    world=AppWorld(
      task_id=label['task_id'],
      experiment_name=label['experiment_name'],
      remote_environment_url=f"http://0.0.0.0:{port}" 
    ) 
    if label['experiment_name'] not in worlds:
      worlds[label['experiment_name']] = {}
    worlds[label['experiment_name']][label['task_id']]=world

  world=worlds[label['experiment_name']][label['task_id']]
  # print("-"*50)
  # print(f"response:{response}")
  if world.task_completed():
    output=None
    print(f"{label['task_id']} has been completed already!")
  else:
    output = world.execute(response)
    print("*"*30)
    #print(f"output:{output}")
    if not world.task_completed():
      print(f"{label['task_id']} remain uncompleted!")
    else:
      print(f"{label['task_id']} from uncompleted to completed!")
    print("*"*30)
  worlds[label['experiment_name']][label['task_id']] =world
  return output

class StepRequest(BaseModel):
    responses: list[str]
    labels: list[dict]

@app.post("/step")
def environment_func(request: StepRequest):
    responses = request.responses
    labels = request.labels
    obs = []
    ps = []
    for response, label in zip(responses, labels):
        if label['experiment_name'] not in ports:
          ports[label['experiment_name']] = {}
        if label['task_id'] not in ports[label['experiment_name']]:
          ports[label['experiment_name']][label['task_id']] = get_next_port()
          ps.append(ports[label['experiment_name']][label['task_id']])
    start_server(ps)
    for response, label in zip(responses, labels):
        obs_single = environment_func_single(response, label)
        obs.append(obs_single)
    
    return obs

class RewardRequest(BaseModel):
    labels: list[dict]

@app.post("/get_reward")
def reward_func(request: RewardRequest):
  rewards=[]
  for label in request.labels:
    test_tracker=evaluate_task(task_id=label['task_id'],experiment_name=label['experiment_name'])
    pass_rate = test_tracker.pass_count / test_tracker.num_tests
    # print(f"{task_id}:After evaluation: Success = {test_tracker.success},pass_rate={pass_rate}")
    # if test_tracker.success:
    #     reward_completion=1
    # else:
    #     reward_completion=0
    # reward_testcases=pass_rate
    # reward={"reward_completion":reward_completion,"reward_testcases":reward_testcases}
    print(f"pass_rate:{pass_rate}")
    rewards.append(pass_rate)
  return {"rewards": rewards}
