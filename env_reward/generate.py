import openai
from prompt import generate_combined_prompts_one 
import backoff

def quota_giveup(e):
    return isinstance(e, openai.error.RateLimitError) and "quota" in str(e)

@backoff.on_exception(
    backoff.constant,
    openai.error.OpenAIError,
    giveup=quota_giveup,
    raise_on_giveup=True,
    interval=20
)

def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    try:
        if engine in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:  # ChatGPT 模型
            result = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "system", "content": "You are an SQL assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            sql = result["choices"][0]["message"]["content"]
        else:  # 旧的文本补全模型
            result = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            sql = 'SELECT' + result["choices"][0]["text"]
    except Exception as e:
        sql = f'error:{e}'
    
    return sql

def collect_oneresponse_from_gpt(db_path, question, api_key, engine, knowledge=None):
    """
    获取单个问题对应的sql查询语句
    
    :param db_path: str - 数据库路径
    :param question: str 
    :param api_key: str 
    :param engine: str 
    :param knowledge: str (optional) - 额外的知识信息
    :return: str - 生成的 SQL 语句
    """
    openai.api_key = api_key

    print('--------------------- Processing Question ---------------------')
    print(f'The question is: {question}')
    
    if knowledge:
        cur_prompt = generate_combined_prompts_one(db_path=db_path, question=question, knowledge=knowledge)
    else:
        cur_prompt = generate_combined_prompts_one(db_path=db_path, question=question)
    
    print(f"-------------cur_prompt: {cur_prompt} -------")
    
    # 连接 GPT 生成 SQL
    plain_result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0, stop=['--', '\n\n', ';', '#'])
    
    # 解析 GPT 生成的结果
    if isinstance(plain_result, str):
        sql = plain_result
    else:
        sql = 'SELECT' + plain_result['choices'][0]['text']
    
    print(f"The answer is: {sql}")
    

    return sql