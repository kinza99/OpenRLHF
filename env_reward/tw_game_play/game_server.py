import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import textworld.gym
from typing import Dict
from threading import Lock
from io import StringIO
import sys

app = FastAPI()

# 存储活跃的游戏实例，每一条对应一个活跃的游戏和对应的状态们
games: Dict[str, Dict] = {}
games_lock = Lock()

class StartGameRequest(BaseModel):
    game_file: str
    max_steps: int = 20

class StepRequest(BaseModel):
    game_id: str
    command: str #模型发送过来的steprequest

@app.post("/start")
def start_game(request: StartGameRequest):
    with games_lock:
        env_id = textworld.gym.register_game(request.game_file, max_episode_steps=request.max_steps)
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()
        game_id = str(uuid.uuid4())

        games[game_id] = {
            "env": env,
            "obs": obs,
            "infos": infos,
            "score": 0,
            "moves": 0,
            "done": False
        }

    return {"game_id": game_id, "observation": obs}

def capture_render_output(env):
    # 使用 StringIO 捕获 render 输出
    old_stdout = sys.stdout  # 保存当前标准输出
    sys.stdout = StringIO()  # 重定向标准输出到 StringIO
    env.render()  # 调用 render()
    output = sys.stdout.getvalue()  # 获取 render 输出内容
    sys.stdout = old_stdout  # 恢复标准输出
    return output

@app.post("/step")
def step(request: StepRequest):
    with games_lock:
        if request.game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found.")

        game = games[request.game_id]

        if game["done"]:
            return {"done": True, "observation": "Game already ended.", "score": game["score"], "moves": game["moves"], "render": ""}

        obs, score, done, infos = game["env"].step(request.command)
        game["obs"] = obs
        game["score"] = score
        game["moves"] += 1
        game["done"] = done

        # 获取 render 输出
        render_output = capture_render_output(game["env"])

    return {
        "observation": obs,
        "score": score,
        "done": done,
        "moves": game["moves"],
        "render": render_output
    }

@app.post("/close")
def close_game(game_id: str):
    with games_lock:
        if game_id in games:
            games[game_id]["env"].close()
            del games[game_id]
            return {"message": f"Game {game_id} closed."}
        else:
            raise HTTPException(status_code=404, detail="Game not found.")

