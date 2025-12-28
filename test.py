import json
import requests
from search.boChaSearch import bocha_websearch_tool

def execute_tool_call(tool_call):
    """执行工具调用并返回结果"""
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    if function_name == "bocha_websearch_tool":
        query = arguments.get("query", "")
        print(f"\n🔍 正在搜索: {query}")
        # 处理StructuredTool对象，使用其run方法
        search_result = bocha_websearch_tool.run({"query": query})
        return search_result
    else:
        return f"未知工具: {function_name}"

def test_direct_api_call():
    """直接调用DeepSeek API，实现工具调用逻辑"""

    # DeepSeek API端点
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {'sk-762ca3eceebf48c1984a178a969d2580'}",
        "Content-Type": "application/json"
    }

    # 初始消息
    messages = [
        {"role": "user", "content": "今天北京天气如何？"}
    ]

    # 构建初始请求
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bocha_websearch_tool",
                    "description": "联网搜索最新信息，必须被调用",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询词"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    }

    print("发送请求数据:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # 第一次API调用
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print("\nAPI响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 检查是否有tool_calls
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0].get('message', {})
            
            # 添加助手回复到消息历史
            messages.append(message)
            
            if 'tool_calls' in message:
                print(f"\n✅ 模型请求调用工具: {message['tool_calls']}")
                
                # 执行每个工具调用
                for tool_call in message['tool_calls']:
                    tool_result = execute_tool_call(tool_call)
                    print(f"\n🔧 工具执行结果: {tool_result[:200]}...")
                    
                    # 将工具结果添加到消息历史
                    messages.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "content": tool_result
                    })
                
                # 第二次API调用，将工具结果发送给模型
                print("\n🔄 发送工具结果给模型...")
                second_response = requests.post(url, headers=headers, json={
                    "model": "deepseek-chat",
                    "messages": messages
                })
                
                if second_response.status_code == 200:
                    final_result = second_response.json()
                    print("\n最终结果:")
                    print(json.dumps(final_result, indent=2, ensure_ascii=False))
                    
                    if 'choices' in final_result and len(final_result['choices']) > 0:
                        final_message = final_result['choices'][0].get('message', {})
                        print(f"\n💬 最终回答: {final_message.get('content', '')}")
                else:
                    print(f"第二次API请求失败: {second_response.status_code}")
                    print(second_response.text)
            else:
                print(f"\n❌ 没有tool_calls字段")
                print(f"完整消息: {message}")
    else:
        print(f"API请求失败: {response.status_code}")
        print(response.text)

if __name__ == '__main__':

    test_direct_api_call()