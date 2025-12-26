import os
from openai import OpenAI

# 设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-762ca3eceebf48c1984a178a969d2580"

# 初始化客户端
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"  # API端点
)

# 简单调用
response = client.chat.completions.create(
    model="deepseek-chat",  # 免费模型
    messages=[
        {"role": "user", "content": "你好！"}
    ],
    stream=False
)

print(response.choices[0].message.content)