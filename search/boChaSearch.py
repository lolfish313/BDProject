import requests
from langchain.tools import tool
import json

BOCHA_API_KEY = "sk-2d843ea3e8bf434da0cdc9ce1c0da035"

@tool
def bocha_websearch_tool(query: str, count: int = 10) -> str:
    """
    使用Bocha Web Search API 进行网页搜索。

    参数:
    - query: 搜索关键词
    - freshness: 搜索的时间范围
    - summary: 是否显示文本摘要
    - count: 返回的搜索结果数量

    返回:
    - 搜索结果的详细信息，包括网页标题、网页URL、网页摘要、网站名称、网站Icon、网页发布时间等。
    """

    # 设置API请求的URL
    url = 'https://api.bochaai.com/v1/web-search'
    # 设置请求头，包含认证信息和内容类型
    headers = {
        'Authorization': f'Bearer {BOCHA_API_KEY}',  # 请替换为你的API密钥
        'Content-Type': 'application/json'
    }
    # 构建请求数据，包含搜索参数
    data = {
        "query": query,  # 搜索关键词
        "freshness": "noLimit",  # 搜索的时间范围，例如 "oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"
        "summary": True,  # 是否返回长文本摘要
        "count": count  # 返回的搜索结果数量
    }
    print(query)

    # 发送POST请求到API
    response = requests.post(url, headers=headers, json=data)

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析JSON响应
        json_response = response.json()
        try:
            # 检查API响应状态和是否有数据
            if json_response["code"] != 200 or not json_response["data"]:
                return f"搜索API请求失败，原因是: {response.msg or '未知错误'}"

            # 获取网页搜索结果
            webpages = json_response["data"]["webPages"]["value"]
            # 检查是否有搜索结果
            if not webpages:
                return "未找到相关结果。"
            # 格式化搜索结果
            formatted_results = ""
            for idx, page in enumerate(webpages, start=1):
                formatted_results += (
                    f"引用: {idx}\n"  # 结果序号
                    f"标题: {page['name']}\n"  # 网页标题
                    f"URL: {page['url']}\n"  # 网页链接
                    f"摘要: {page['summary']}\n"  # 网页摘要
                    f"网站名称: {page['siteName']}\n"  # 网站名称
                    f"网站图标: {page['siteIcon']}\n"  # 网站图标链接
                    f"发布时间: {page['dateLastCrawled']}\n\n"  # 最后抓取时间
                )
                print(formatted_results)
            return formatted_results.strip()  # 返回格式化后的结果，去除首尾空白
        except Exception as e:
            # 处理解析搜索结果时的异常
            return f"搜索API请求失败，原因是：搜索结果解析失败 {str(e)}"
    else:
        # 处理API请求失败的情况
        return f"搜索API请求失败，状态码: {response.status_code}, 错误信息: {response.text}"


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

if __name__ == '__main__':
    # 使用工具对象的run方法来执行搜索
    print(bocha_websearch_tool.run({"query": "AK104 宫颈癌 中国NMPA治疗获批情况"}))
