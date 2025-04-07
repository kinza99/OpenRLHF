import logging
from openai import OpenAI
import re
from jinja2 import Template
from appworld.task import Task
from appworld import evaluate_task
from appworld import AppWorld

import os



# This is a basic prompt template containing all the necessary onboarding information to solve AppWorld tasks. It explains the role of the agent and the supervisor, how to explore the API documentation, how to operate the interactive coding environment and call APIs via a simple task, and provides key instructions and disclaimers.

# You can adapt it as needed by your agent. You can also choose to bypass API docs app and build your own API retrieval, e.g., for FullCodeRefl, IPFunCall, etc, we asked an LLM to predict relevant APIs separately and put its documentation directly in the prompt.
PROMPT_TEMPLATE = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

For example, consider the task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

What is the password for my Spotify account?

ASSISTANT:
# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

USER:
[
  {
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  },
  {
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  },
  ...
  {
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  },
  {
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  },
  ...
]


ASSISTANT:
# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))


USER:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]


ASSISTANT:
# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

USER:
{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {
    'success': [{'account_name': 'string', 'password': 'string'}],
    'failure': {'message': 'string'}
  }
}

ASSISTANT:
# Okay, it requires no arguments. So I can just call it directly.
print(apis.supervisor.show_account_passwords())

USER:
[
  {
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  },
  {
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  },
  ...
]


ASSISTANT:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)


USER:
dummy_spotify_pass

ASSISTANT:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)


USER:
Marked the active task complete.


----------------------------------------------

USER:
**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current data or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

{{ instruction }}
"""


# logging.basicConfig(
#     filename='task_execution_miniagent_train_deepseek_trysplitexereward.log',  
#     level=logging.INFO,  
#     format='%(asctime)s - %(levelname)s - %(message)s',  
# )

client = OpenAI(api_key="", base_url="https://api.deepseek.com")
model = "deepseek-chat"

def call_llm(messages: list[dict]) -> str:
    """
    Call an LLM with a history of messages and return the response.
    """
    # Change this function to modify the base llm
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=400, seed=123
    )
    text_response = ""
    if response.choices:
        text_response = response.choices[0].message.content
    return text_response

class MinimalReactAgent:
    """A minimal ReAct Agent for AppWorld tasks."""

    def __init__(self, task: Task):
        self.task = task
        self.history: list[dict] = self.prompt_messages()

    def prompt_messages(self) -> list[dict]:
        """Builds prompt messages for the agent to solve self.task.instruction"""
        dictionary = {"supervisor": self.task.supervisor, "instruction": self.task.instruction}
        prompt = Template(PROMPT_TEMPLATE.lstrip()).render(dictionary)
        messages: list[dict] = []
        last_start = 0
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
            last_end = match.span()[0]
            if len(messages) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {prompt[:last_end]}"
                    )
            else:
                messages[-1]["content"] = prompt[last_start:last_end]
            mesg_type = match.group(1).lower()
            messages.append({"role": mesg_type, "content": None})
            last_start = match.span()[1]
        messages[-1]["content"] = prompt[last_start:]
        return messages

    def next_code_block(self, last_execution_output: str | None = None) -> str:
        """
        Asks Agent to generate next code block given last_execution_output and history.
        """
        if last_execution_output is not None:
            self.history.append({"role": "user", "content": last_execution_output})
        # Get the next code block based on the history.
        code = call_llm(self.history)
        # Add this code block to history
        self.history.append({"role": "assistant", "content": code})
        return code


worlds={} #worlds是字典类型所以可以直接通过实验名称和对应的taskid找到对应的world
max_interactions=50

def exe(labels):
    create_games(labels)
    obs = [{'output': '', 'iscompleted': False} for _ in range(len(labels))]
    for _ in range(max_interactions):
        alldone=True
        responses=collect_responses(labels,obs)
        obs=environment_func(responses,labels)
        for single_obs in obs:
          if single_obs['iscompleted']==False:
            alldone=False
            break
        if alldone==True: 
          break

def create_games(labels):
  for label in labels:
    if label['experiment_name'] not in worlds:
        worlds[label['experiment_name']] = {}
    world=AppWorld(
      task_id=label['task_id'],
      experiment_name=label['experiment_name'],
    ) 
    agent = MinimalReactAgent(world.task)
    worlds[label['experiment_name']][label['task_id']]={'world':world,'agent':agent} #每个world分配一个agent
  return worlds        

def collect_responses(labels, obs):
    responses = []
    for index, label in enumerate(labels):
        agent = worlds[label['experiment_name']][label['task_id']]['agent']
        if obs[index]['iscompleted'] == True:
            response = ""
        else:
            response = agent.next_code_block(obs[index]['output'])
        responses.append(response)
    return responses


def environment_func( responses, labels, queries=None):
  obs = []
  for response, label in zip(responses, labels):
    print(f"{label['task_id']} in environment_func!")
    print(f"current response is {response}")
    world=worlds[label['experiment_name']][label['task_id']]['world']
    # output =world.execute(response)
    # print(F"Output is :{output}.")
    if world.task_completed():
      print(f"{label['task_id']} ended!")
      iscompleted=True
      output=""
    else:
      iscompleted=False
      output =world.execute(response)
    obs.append({'output':output,'iscompleted':iscompleted})
  return obs

def reward_func(labels,queries=None, responses=None):
  rewards=[]
  for label in labels:
    test_tracker=evaluate_task(task_id=label['task_id'],experiment_name=label['experiment_name'])
    pass_rate = test_tracker.pass_count / test_tracker.num_tests
    # print(f"{task_id}:After evaluation: Success = {test_tracker.success},pass_rate={pass_rate}")
    if test_tracker.success:
        reward_completion=1
    else:
        reward_completion=0
    reward_testcases=pass_rate
    reward={"reward_completion":reward_completion,"reward_testcases":reward_testcases}
    rewards.append(reward)
  return rewards

def main(labels):
  exe(labels)
  rewards=reward_func(labels)
  print(f"rewards:{rewards}")


# if __name__=="__main__":
#   experiment_name = "minimal_react_agent_train_deepseek_trysplitexereward"
#   labels=[{'experiment_name':experiment_name,'task_id':"82e2fac_1"},{'experiment_name':experiment_name,'task_id':"82e2fac_2"},{'experiment_name':experiment_name,'task_id':"82e2fac_3"}]
#   main(labels)

import json

def read_task_ids_from_jsonl(file_path, num_lines=5):
    task_ids = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= num_lines:
                break
            data = json.loads(line) 
            task_ids.append(data['label']['task_id']) 
    return task_ids

if __name__ == "__main__":
    experiment_name = "minimal_react_agent_train_deepseek_trysplitexereward"

    file_path = "/cpfs01/shared/llm_code/hesiyang/appworld_new/data/appworld.jsonl"
    task_ids = read_task_ids_from_jsonl(file_path)
    
    labels = [{'experiment_name': experiment_name, 'task_id': task_id} for task_id in task_ids]
    
    # Call main with the labels
    main(labels)