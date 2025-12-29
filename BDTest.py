import sys
import os
import time
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Any
from datetime import datetime
from search.boChaSearch import bocha_websearch_tool

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ChatOpenAI import ChatOpenAIModel

model_name = "deepseek-chat"
# 初始化LLM
llm = ChatOpenAIModel.chatOpenAI(model=model_name,
                                 base_url="https://api.deepseek.com",
                                 api_key='sk-762ca3eceebf48c1984a178a969d2580')
llm = llm.bind_tools([bocha_websearch_tool],tool_choice={"type": "function", "function": {"name": "bocha_websearch_tool"}})

def checkIfUseTools(response):
    # 检查是否调用了工具
    if response.tool_calls:
        print("模型要求调用工具!")
        for tool_call in response.tool_calls:
            print(f"工具名: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
            # 执行工具调用
            if tool_call['name'] == 'bocha_websearch_tool':
                tool_result = bocha_websearch_tool.run(tool_call['args'])
                print(f"工具执行结果: {len(tool_result)}")
                return tool_result
    return None
# 定义状态类型
class BDAnalysisState(TypedDict):
    user_input: str
    drug_info: Dict[str, Any]
    market_analysis: str
    drug_analysis: str
    institution_analysis: str
    risk_check_market: str
    risk_check_drug: str
    risk_check_institution: str
    bd_analysis: str
    risk_check_final: str
    final_report: str


def info_integrator(state: BDAnalysisState) -> BDAnalysisState:
    """基础信息整合专员：解析用户输入为结构化信息"""
    prompt = f"""
    你是一位专业的创新药BD基础信息整合专员。请根据用户输入，提取并结构化以下信息：

    用户输入：{state['user_input']}

    请提取以下结构化信息，如有多个请返回多个Json，每个药品的Json内容包含：
        1. 创新药药品名称
        2. MoA(Mechanism of Action)
        3. 分子类型
        4. 主适应症（五个以内）
        5. 各适应症的中国NMPA治疗获批（一线/二线/三线）
        6. 中国药物临床期数和临床效果
        7. 美国FDA对药物的态度：积极/中立/消极
        8. 转让方：创新药所在公司
        9. 受让方：如果用户未明确，则需要列出所有潜在的海外受让方列表
        10. 竞品药物列表：
	        1. 同作用机制的药品
	        2. 不同作用机制，但相同适应症的药品

    请以JSON格式返回结果，包含上述所有字段，因为回答将直接转为json，因此不要有其余多余的描述，也无需在json中引用。
    """

    response = llm.invoke(prompt)
    tool_result = checkIfUseTools(response)
    if tool_result:
        # 如果有工具调用结果，将结果添加到prompt中再次请求模型
        prompt_with_tool_result = f"{prompt}\n\n工具搜索结果:\n{tool_result}\n\n请基于以上搜索结果，重新生成JSON格式的药品信息。"
        response = llm.invoke(prompt_with_tool_result)
    print("大模型response : {}".format(response.content))
    # 解析JSON响应并存储到状态中
    try:
        import json
        drug_info = json.loads(response.content)
    except:
        drug_info = {"error": "解析失败", "raw_response": response.content}

    return {**state, "drug_info": drug_info}


def market_analyst(state: BDAnalysisState) -> BDAnalysisState:
    """市场调研分析师：分析适应症的海外市场情况"""
    prompt = f"""
    你是一位专业的医药市场调研分析师。请基于以下创新药信息进行市场分析：

    创新药信息：
    {state['drug_info']}

    请重点分析以下指标：

    ## 1. 当前美国市场规模分析
    使用模型：适应症市场规模 = 目标患者数(P) × 年治疗费用(ATC) × 渗透率（PR）
    - 目标患者数(P) = 适应症全球总患病人数 × 适应症适合治疗率
    - 年治疗费用(ATC) = 当前单次药品平均价格 × 年平均治疗次数
    - 渗透率（PR) = 多少比例患者适合使用药品治疗，需主要参考药物同治疗机制的竞品市场渗透率进行保守预估

    ## 2. 需求紧迫性分析
    - 市场除创新药本身外，患者是否有药可医？
    - 当前适应症的致死率、致残率、对患者生活水平的影响（可参考竞品药物的HRQOL评分等）
    - 已上市的竞品药品治疗效果如何？与创新药预期疗效的比较
    - 患者平均年龄，在50岁以下的占比

    请提供详细的分析报告，包含数据来源说明和推理过程，无需署名。
    """

    response = llm.invoke(prompt)
    return {**state, "market_analysis": response.content}


def drug_analyst(state: BDAnalysisState) -> BDAnalysisState:
    """药物分析师：分析药品本身特征"""
    prompt = f"""
    你是一位专业的药物分析师。请基于以下创新药信息进行药物分析：

    创新药信息：
    {state['drug_info']}

    请重点分析以下指标：

    ## 1. 药物全球潜力分析
    - 药物全球顺位：是否为Global First-in-Class (FIC) 或 Best-in-Class (BIC)？(如果产业已有较多临床依据，则以临床依据比较为主，无需过多参考国内卖方研报等信息)
    - 全球竞品情况：现有竞品的数量和疗效比较。涉及到与同机制药物竞品、现有标准疗法药品、在研具备较好疗效的潜在竞品的比较，比较范围可以是相同适应症，但不同机制的药物。
    - 中国市场准入情况：是否纳入中国国家医保或商业保险
    - 中国临床期数，临床效果如何？
    - 中国NMPA治疗获批（一线/二线/三线）情况，若获批一线治疗，则说明药物相对较高的潜力。

    ## 2. 药物海外PoC分析
    - FDA对药物的态度：积极/中立/消极
    - 药物海外临床期数、临床效果如何？
    - 是否有硬性指标说明药物有效性？
    - 是否有药物相关的论文研究支持？（需要考虑论文发表的杂志水准综合判断）
    - 同类药物是否有海外PoC，若有，其效果如何？

    请提供详细的分析报告，包含数据来源说明，无需署名。
    """

    response = llm.invoke(prompt)
    return {**state, "drug_analysis": response.content}


def institution_analyst(state: BDAnalysisState) -> BDAnalysisState:
    """机构分析师：分析买卖双方机构特征"""
    prompt = f"""
    你是一位专业的机构分析师。请基于以下创新药信息进行机构分析：

    创新药信息：
    {state['drug_info']}

    请重点分析以下指标：

     1. 转让方Track Record分析
    - 转让机构历史上是否有创新药交易？成功/失败记录？
    - 转让机构历史BD交易的规模、频率

    2. 受让方近期BD动力分析（如果没有明确受让方，需要分析潜在受让方画像），需主要分析海外受让方
    - 受让方是否有药品专利即将到期？
    - 受让方是否有相同适应症领域的药品即将到期？
    - 受让方是否已经布局同适应症的其他药物？
    - 即将到期药品专利的市值规模？
    - 受让方对适应症领域的药品市场是否看好？

    请提供详细的分析报告，包含数据来源说明，无需署名。"""
    response = llm.invoke(prompt)
    return {**state, "institution_analysis": response.content}


def risk_checker_market(state: BDAnalysisState) -> BDAnalysisState:
    """风控专员：检查市场分析报告的适当性和准确性"""
    prompt = f"""
    你是一位专业的医药风控专员。请基于创新药基本信息，检查以下市场分析报告：

    创新药基本信息：
    {state['drug_info']}

    市场分析报告：
    {state['market_analysis']}

    检查标准：
    1. 数据准确性：数据是否有可靠来源和依据？
    2. 内容适当性：是否包含"一定会发生BD"等绝对性描述？
    3. 合规性：是否符合医药行业报告标准？
    4. 一致性：市场分析报告里的创新药相关信息与创新药基本信息需要保持一致性

    请给出检查结果：
    - 如果通过检查，返回"通过"
    - 如果未通过，指出具体问题并返回"不通过：具体问题描述"

    报告无需署名
    """
    response = llm.invoke(prompt)
    return {**state, "risk_check_market": response.content}


def risk_checker_drug(state: BDAnalysisState) -> BDAnalysisState:
    """风控专员：检查药物分析报告的适当性和准确性"""
    prompt = f"""
    你是一位专业的医药风控专员。请基于创新药基本信息，检查以下药物分析报告：

    创新药基本信息：
    {state['drug_info']}

    药物分析报告：
    {state['drug_analysis']}

    检查标准：
    1. 数据准确性：数据是否有可靠来源和依据？
    2. 内容适当性：是否包含"一定会发生BD"等绝对性描述？
    3. 合规性：是否符合医药行业报告标准？
    4. 一致性：药物分析报告里的创新药相关信息与创新药基本信息需要保持一致性

    请给出检查结果：
    - 如果通过检查，返回"通过"
    - 如果未通过，指出具体问题并返回"不通过：具体问题描述"

    报告无需署名
    """

    response = llm.invoke(prompt)
    return {**state, "risk_check_drug": response.content}


def risk_checker_institution(state: BDAnalysisState) -> BDAnalysisState:
    """风控专员：检查机构分析报告的适当性和准确性"""
    prompt = f"""
    你是一位专业的医药风控专员。请基于创新药基本信息，检查以下机构分析报告：

    创新药基本信息：
    {state['drug_info']}

    机构分析报告：
    {state['institution_analysis']}

    检查标准：
    1. 数据准确性：数据是否有可靠来源和依据？
    2. 内容适当性：是否包含"一定会发生BD"等绝对性描述？
    3. 合规性：是否符合医药行业报告标准？
    4. 一致性：机构分析报告里的创新药相关信息与创新药基本信息需要保持一致性

    请给出检查结果：
    - 如果通过检查，返回"通过"
    - 如果未通过，指出具体问题并返回"不通过：具体问题描述"

    报告无需署名
    """

    response = llm.invoke(prompt)
    return {**state, "risk_check_institution": response.content}


def bd_analysis_specialist(state: BDAnalysisState) -> BDAnalysisState:
    """BD潜力业务分析专员：整合分析结果，评估BD潜力"""
    prompt = f"""
    你是一位专业的BD潜力业务分析专员。请基于以下分析报告进行综合评估：

    市场分析报告：
    {state['market_analysis']}

    药物分析报告：
    {state['drug_analysis']}

    机构分析报告：
    {state['institution_analysis']}

    请进行以下分析：
    1. 对三位分析师报告，分别进行简短总结
    2. 判断BD潜力等级：极高/高/中高/中/低/极低
    2. 给出判断理由
    3. 如涉及多个创新药，请进行BD潜力排序

    请提供详细的BD潜力分析报告，报告无需署名。
    """

    response = llm.invoke(prompt)
    return {**state, "bd_analysis": response.content}


def risk_checker_final(state: BDAnalysisState) -> BDAnalysisState:
    """风控专员：检查最终BD分析报告的适当性和准确性"""
    prompt = f"""
    你是一位专业的医药风控专员。请基于创新药基本信息，检查以下最终BD分析报告：

    创新药基本信息：
    {state['drug_info']}

    BD分析报告：
    {state['bd_analysis']}

    检查标准：
    1. 数据准确性：结论是否有可靠的数据支持？
    2. 内容适当性：是否包含"一定会发生BD"等绝对性描述？
    3. 逻辑合理性：分析推理是否合理？
    4. 合规性：是否符合医药行业报告标准？
    5. 一致性：BD分析报告里的创新药相关信息与创新药基本信息需要保持一致性

    请给出检查结果：
    - 如果通过检查，返回"通过"
    - 如果未通过，指出具体问题并返回"不通过：具体问题描述"
    """

    response = llm.invoke(prompt)
    return {**state, "risk_check_final": response.content}


def final_report_generator(state: BDAnalysisState) -> BDAnalysisState:
    """生成最终的Markdown格式分析报告"""
    current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # 检查风控结果
    risk_status = {
        "市场分析": state['risk_check_market'],
        "药物分析": state['risk_check_drug'],
        "机构分析": state['risk_check_institution'],
        "最终分析": state['risk_check_final']
    }

    markdown_content = \
f"""# 创新药BD潜力分析报告
    
## 1. 基础信息整合
```json
{state['drug_info']}
```
    
## 2. 市场调研分析报告
{state['market_analysis']}

## 3. 药物分析报告  
{state['drug_analysis']}

## 4. 机构分析报告
{state['institution_analysis']}

## 5. BD潜力综合分析
{state['bd_analysis']}

"""

    return {**state, "final_report": markdown_content}


def save_to_markdown(state: BDAnalysisState) -> BDAnalysisState:
    """保存Markdown文件"""
    desktop_path = os.path.expanduser("~/Desktop/BDProject/BDProject")
    filename = f"BD潜力分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = os.path.join(desktop_path, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(state['final_report'])
        print(f"BD潜力分析报告已保存到：{filepath}")
    except Exception as e:
        print(f"保存文件时出错：{e}")

    return state


# 条件路由函数
def should_rerun_market_analysis(state: BDAnalysisState) -> str:
    """判断是否需要重新进行市场分析"""
    if "不通过" in state['risk_check_market']:
        return "market_analyst"
    return "drug_analyst"


def should_rerun_drug_analysis(state: BDAnalysisState) -> str:
    """判断是否需要重新进行药物分析"""
    if "不通过" in state['risk_check_drug']:
        return "drug_analyst"
    return "institution_analyst"


def should_rerun_institution_analysis(state: BDAnalysisState) -> str:
    """判断是否需要重新进行机构分析"""
    if "不通过" in state['risk_check_institution']:
        return "institution_analyst"
    return "bd_analysis_specialist"


def should_rerun_final_analysis(state: BDAnalysisState) -> str:
    """判断是否需要重新进行最终分析"""
    if "不通过" in state['risk_check_final']:
        return "bd_analysis_specialist"
    return "final_report_generator"


# 构建LangGraph工作流
def create_bd_analysis_agent():
    workflow = StateGraph(BDAnalysisState)

    # 添加节点
    workflow.add_node("info_integrator", info_integrator)
    workflow.add_node("market_analyst", market_analyst)
    workflow.add_node("drug_analyst", drug_analyst)
    workflow.add_node("institution_analyst", institution_analyst)
    workflow.add_node("risk_checker_market", risk_checker_market)
    workflow.add_node("risk_checker_drug", risk_checker_drug)
    workflow.add_node("risk_checker_institution", risk_checker_institution)
    workflow.add_node("bd_analysis_specialist", bd_analysis_specialist)
    workflow.add_node("risk_checker_final", risk_checker_final)
    workflow.add_node("final_report_generator", final_report_generator)
    workflow.add_node("save_markdown", save_to_markdown)

    # 设置工作流路径
    workflow.set_entry_point("info_integrator")

    # 基础流程
    workflow.add_edge("info_integrator", "market_analyst")
    workflow.add_conditional_edges("market_analyst", should_rerun_market_analysis, {
        "market_analyst": "market_analyst",
        "drug_analyst": "risk_checker_market"
    })
    workflow.add_edge("risk_checker_market", "drug_analyst")
    workflow.add_conditional_edges("drug_analyst", should_rerun_drug_analysis, {
        "drug_analyst": "drug_analyst",
        "institution_analyst": "risk_checker_drug"
    })
    workflow.add_edge("risk_checker_drug", "institution_analyst")
    workflow.add_conditional_edges("institution_analyst", should_rerun_institution_analysis, {
        "institution_analyst": "institution_analyst",
        "bd_analysis_specialist": "risk_checker_institution"
    })
    workflow.add_edge("risk_checker_institution", "bd_analysis_specialist")
    workflow.add_conditional_edges("bd_analysis_specialist", should_rerun_final_analysis, {
        "bd_analysis_specialist": "bd_analysis_specialist",
        "final_report_generator": "risk_checker_final"
    })
    workflow.add_edge("risk_checker_final", "final_report_generator")
    workflow.add_edge("final_report_generator", "save_markdown")
    workflow.add_edge("save_markdown", END)

    return workflow.compile()

def create_bd_analysis_agent_test():
    workflow = StateGraph(BDAnalysisState)

    # 添加节点
    workflow.add_node("info_integrator", info_integrator)
    workflow.add_node("save_markdown", save_to_markdown)

    # 设置工作流路径
    workflow.set_entry_point("info_integrator")

    # 基础流程
    workflow.add_edge("info_integrator", "save_markdown")
    workflow.add_edge("save_markdown", END)

    return workflow.compile()


# 主函数：运行BD分析助手
def run_bd_analysis(user_input: str):
    """运行BD分析助手的主函数"""
    # 初始化状态
    initial_state = BDAnalysisState(
        user_input=user_input,
        drug_info={},
        market_analysis="",
        drug_analysis="",
        institution_analysis="",
        risk_check_market="",
        risk_check_drug="",
        risk_check_institution="",
        bd_analysis="",
        risk_check_final="",
        final_report=""
    )

    # 创建并运行工作流
    bd_agent = create_bd_analysis_agent_test()
    result = bd_agent.invoke(initial_state)

    return result

# 示例使用
if __name__ == "__main__":
    # 示例查询
    sample_query = "分析AK104的BD潜力"

    print("🚀 开始创新药BD潜力分析...")
    start_time = time.time()
    result = run_bd_analysis(sample_query)
    end_time = time.time()
    execution_time = end_time - start_time

    print("✅ BD潜力分析完成！")
    print(f"⏱️  总耗时：{execution_time:.2f} 秒")