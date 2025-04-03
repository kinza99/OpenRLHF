from openai import OpenAI
import requests
import time
import re

# === Configuration ===
OPENAI_API_KEY =""
TEXTWORLD_SERVER_URL = "http://localhost:8000" 
#GAME_FILE_PATH = "/home/hesiyang/tw_games/adventure_game.z8"
client = OpenAI(api_key=OPENAI_API_KEY)


def create_game(GAME_FILE_PATH):
    response = requests.post(f"{TEXTWORLD_SERVER_URL}/start", json={
        "game_file": GAME_FILE_PATH,
        "max_steps": 20
    })
    response.raise_for_status()
    return response.json()

def step_game(game_id, command):
    response = requests.post(f"{TEXTWORLD_SERVER_URL}/step", json={
        "game_id": game_id,
        "command": command
    })
    response.raise_for_status()
    return response.json()

def close_game(game_id):
    requests.post(f"{TEXTWORLD_SERVER_URL}/close", params={"game_id": game_id})

def generate_command_with_gpt(history):
    prompt = "\n".join(history + ["Based on the above information, provide the next action command, don't explain, just output the command:"])

    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are an AI that excels at playing text adventure games. Only return the command."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

def complete_one_game(GAME_FILE_PATH):

    game = create_game(GAME_FILE_PATH)
    game_id = game["game_id"]
    print(f"New game started, Game ID: {game_id}\n")

    history = [f"Initial observation: {game['observation']}"]

    done = False
    while not done:
        try:
            command = generate_command_with_gpt(history)
            print(f"GPT generated command: {command}")

            result = step_game(game_id, command)

            print(f"[STEP {result['moves']}]")
            print("Render Output:\n", result["render"])
            #print("Observation:", result["observation"])
            #print("Score:", result["score"])
            print("=" * 50)

            # Update history
            history.append(f"Game feedback: {result['render']}")
            history.append(f"My input: {command}")

            done = result["done"]
            time.sleep(1)

        except Exception as e:
            print("Error occurred:", e)
            break

    close_game(game_id)
    print("Game over, closed the game instance.")
    print("*"*50)
    print(f"Completion_score is {result['score']}.")
    print("*"*50)
    return result


def exe_reward_compute(GAME_FILE_PATH):
    match = re.search(r'game_\d+_\d+_\d+_(\d+)_\d+', GAME_FILE_PATH)
    moves_gt=int(match.group(1))
    result=complete_one_game(GAME_FILE_PATH)
    completion_reward=result['score']
    if completion_reward==2:
        final_moves=result['moves']
        efficiency_reward=moves_gt/final_moves
    else:
        efficiency_reward=0

    return {'completion_reward':completion_reward,'efficiency_rewrd':efficiency_reward}


# def main():
#     complete_one_game(GAME_FILE_PATH)
# if __name__ == "__main__":
#     main()

