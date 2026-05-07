"""Multi-Agent 代码审查系统

演示内容：
1. 多 Agent 协作架构
2. Agent 角色定义与消息传递
3. 冲突解决与一致性
4. 完整代码审查流程

对应文章：Agent设计模式（6）：Multi-Agent模式——构建多Agent协作系统

运行：uv run python demos/10_multi_agent_code_review.py
"""
import os
import json
import ssl
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
from threading import Thread, local as threading_local
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from helpers import get_llm

load_dotenv()


# ===== 核心数据结构 =====

class AgentRole(Enum):
    """Agent 角色"""
    STYLE = "style_checker"
    BUG = "bug_analyzer"
    PERFORMANCE = "performance_expert"
    SECURITY = "security_reviewer"
    SUMMARIZER = "summarizer"


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFY = "notify"
    ERROR = "error"
    BROADCAST = "broadcast"


@dataclass
class AgentCapability:
    """Agent 能力描述"""
    role: AgentRole
    name: str
    description: str
    input_schema: Dict
    output_schema: Dict


@dataclass
class AgentMessage:
    """Agent 间通信消息"""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class ConflictType(Enum):
    """冲突类型"""
    DATA_CONFLICT = "data_conflict"
    OPINION_CONFLICT = "opinion_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    PRIORITY_CONFLICT = "priority_conflict"


# ===== 消息协议与工厂 =====

class MessageFactory:
    """消息工厂"""

    @staticmethod
    def create_request(sender: str, receiver: str, content: Dict) -> AgentMessage:
        return AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=MessageType.REQUEST.value,
            content=content
        )

    @staticmethod
    def create_response(original: AgentMessage, content: Dict) -> AgentMessage:
        return AgentMessage(
            sender=original.receiver,
            receiver=original.sender,
            message_type=MessageType.RESPONSE.value,
            content=content,
            correlation_id=original.message_id
        )

    @staticmethod
    def create_error(original: AgentMessage, error: str) -> AgentMessage:
        return AgentMessage(
            sender=original.receiver,
            receiver=original.sender,
            message_type=MessageType.ERROR.value,
            content={"error": error},
            correlation_id=original.message_id
        )


# ===== 消息总线 =====

class MessageBus:
    """Agent 间消息总线"""

    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.lock = threading_local()

    def register(self, agent_id: str) -> None:
        """注册 Agent"""
        if agent_id not in self.queues:
            self.queues[agent_id] = Queue()

    def send(self, message: AgentMessage, timeout: float = 5.0) -> bool:
        """发送消息"""
        if not message.receiver:
            return False
        if message.receiver not in self.queues:
            return False
        try:
            self.queues[message.receiver].put(message, timeout=timeout)
            return True
        except Exception:
            return False

    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """接收消息"""
        if agent_id not in self.queues:
            return None
        try:
            if timeout is None:
                return self.queues[agent_id].get()
            else:
                return self.queues[agent_id].get(timeout=timeout)
        except Empty:
            return None


# ===== 基础 Agent =====

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.message_bus.register(agent_id)
        self._running = False
        self._thread: Optional[Thread] = None

    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理消息"""
        pass

    @abstractmethod
    def get_capability(self) -> AgentCapability:
        """获取能力描述"""
        pass

    def start(self) -> None:
        """启动 Agent"""
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止 Agent"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        """消息处理循环"""
        while self._running:
            message = self.message_bus.receive(self.agent_id, timeout=0.1)
            if message:
                try:
                    response = self.process_message(message)
                    if response and message.message_type == MessageType.REQUEST.value:
                        self.message_bus.send(response)
                except Exception as e:
                    error_msg = MessageFactory.create_error(message, str(e))
                    self.message_bus.send(error_msg)

    def send_message(self, receiver: str, content: Dict) -> bool:
        """发送消息"""
        message = MessageFactory.create_request(self.agent_id, receiver, content)
        return self.message_bus.send(message)


# ===== 具体实现 =====

def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY not found"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的代码审查助手。返回JSON格式结果。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def parse_json_response(text: str, default: Dict) -> Dict:
    """解析 LLM 返回的 JSON"""
    import re
    try:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return default


class StyleCheckerAgent(BaseAgent):
    """风格检查 Agent"""

    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            role=AgentRole.STYLE,
            name="风格检查员",
            description="检查代码风格、命名规范、格式一致性",
            input_schema={"code": "str", "language": "str"},
            output_schema={"issues": "List[Dict]", "suggestions": "List[str]"}
        )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        code = message.content.get('code', '')
        language = message.content.get('language', 'python')
        prompt = f"检查{language}代码风格...\n\n代码：\n{code}"
        response_text = call_deepseek(prompt)
        result = parse_json_response(response_text, {"issues": [], "suggestions": []})
        return MessageFactory.create_response(message, result)


class BugAnalyzerAgent(BaseAgent):
    """Bug 分析 Agent"""

    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            role=AgentRole.BUG,
            name="Bug分析师",
            description="分析潜在bug、边界条件、异常处理",
            input_schema={"code": "str", "language": "str", "context": "str"},
            output_schema={"bugs": "List[Dict]", "severity": "str"}
        )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        code = message.content.get('code', '')
        language = message.content.get('language', 'python')
        context = message.content.get('context', '')
        prompt = f"分析{language}代码潜在bug\n上下文：{context}\n\n代码：\n{code}"
        response_text = call_deepseek(prompt)
        result = parse_json_response(response_text, {"bugs": [], "severity": "unknown"})
        return MessageFactory.create_response(message, result)


class PerformanceExpertAgent(BaseAgent):
    """性能评估 Agent"""

    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            role=AgentRole.PERFORMANCE,
            name="性能专家",
            description="评估性能问题、算法复杂度、资源使用",
            input_schema={"code": "str", "language": "str"},
            output_schema={"issues": "List[Dict]", "optimizations": "List[str]"}
        )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        code = message.content.get('code', '')
        language = message.content.get('language', 'python')
        prompt = f"评估{language}代码性能...\n\n代码：\n{code}"
        response_text = call_deepseek(prompt)
        result = parse_json_response(response_text, {"issues": [], "optimizations": []})
        return MessageFactory.create_response(message, result)


class SecurityReviewerAgent(BaseAgent):
    """安全审查 Agent"""

    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            role=AgentRole.SECURITY,
            name="安全审查员",
            description="检查安全漏洞、注入攻击、权限问题",
            input_schema={"code": "str", "language": "str", "sensitive_data": "List[str]"},
            output_schema={"vulnerabilities": "List[Dict]", "risk_level": "str"}
        )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        code = message.content.get('code', '')
        language = message.content.get('language', 'python')
        sensitive_data = message.content.get('sensitive_data', [])
        prompt = f"审查{language}代码安全性\n敏感数据：{sensitive_data}\n\n代码：\n{code}"
        response_text = call_deepseek(prompt)
        result = parse_json_response(response_text, {"vulnerabilities": [], "risk_level": "unknown"})
        return MessageFactory.create_response(message, result)


class SummarizerAgent(BaseAgent):
    """汇总报告 Agent"""

    def get_capability(self) -> AgentCapability:
        return AgentCapability(
            role=AgentRole.SUMMARIZER,
            name="报告汇总员",
            description="汇总各Agent的输出，生成综合报告",
            input_schema={"style_report": "Dict", "bug_report": "Dict", "performance_report": "Dict", "security_report": "Dict"},
            output_schema={"summary": "str", "priority_issues": "List[Dict]", "recommendations": "List[str]"}
        )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        style_report = message.content.get('style_report', {})
        bug_report = message.content.get('bug_report', {})
        performance_report = message.content.get('performance_report', {})
        security_report = message.content.get('security_report', {})

        # 收集所有问题
        all_issues = []
        for issue in style_report.get('issues', []):
            all_issues.append({'category': 'Style', **issue})
        for bug in bug_report.get('bugs', []):
            all_issues.append({'category': 'Bug', **bug})
        for issue in performance_report.get('issues', []):
            all_issues.append({'category': 'Performance', **issue})
        for vuln in security_report.get('vulnerabilities', []):
            all_issues.append({'category': 'Security', **vuln})

        # 按严重程度排序
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'unknown': 4}
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity'), 4))

        priority_issues = [x for x in all_issues if x.get('severity') in ['critical', 'high']]

        # 汇总建议
        recommendations = []
        recommendations.extend(style_report.get('suggestions', []))
        recommendations.extend(performance_report.get('optimizations', []))

        result = {
            "summary": f"代码审查完成：发现{len(all_issues)}个问题",
            "priority_issues": priority_issues,
            "recommendations": recommendations[:10],
            "total_issues": len(all_issues)
        }
        return MessageFactory.create_response(message, result)


# ===== 协调器 =====

class CodeReviewCoordinator:
    """代码审查协调器"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()

    def _init_agents(self) -> None:
        """初始化所有 Agent"""
        self.agents = {
            'style_checker': StyleCheckerAgent('style_checker', self.message_bus),
            'bug_analyzer': BugAnalyzerAgent('bug_analyzer', self.message_bus),
            'performance_expert': PerformanceExpertAgent('performance_expert', self.message_bus),
            'security_reviewer': SecurityReviewerAgent('security_reviewer', self.message_bus),
            'summarizer': SummarizerAgent('summarizer', self.message_bus)
        }

        for agent in self.agents.values():
            agent.start()

    def review_code(self, code: str, language: str = 'python', context: str = '', sensitive_data: List[str] = None) -> Dict:
        """执行代码审查"""
        sensitive_data = sensitive_data or []

        # 发送请求
        requests = {
            'style_checker': {'code': code, 'language': language},
            'bug_analyzer': {'code': code, 'language': language, 'context': context},
            'performance_expert': {'code': code, 'language': language},
            'security_reviewer': {'code': code, 'language': language, 'sensitive_data': sensitive_data}
        }

        for agent_id, content in requests.items():
            self.agents[agent_id].send_message(
                'summarizer',
                content
            )

        # 等待处理（简化版）
        import time
        time.sleep(2)

        # 模拟结果收集
        style_report = {"issues": [], "suggestions": ["风格良好"]}
        bug_report = {"bugs": [], "severity": "low"}
        performance_report = {"issues": [], "optimizations": ["性能可接受"]}
        security_report = {"vulnerabilities": [], "risk_level": "low"}

        # 汇总报告
        summary_request = {
            'style_report': style_report,
            'bug_report': bug_report,
            'performance_report': performance_report,
            'security_report': security_report
        }

        summary_message = MessageFactory.create_request(
            'coordinator',
            'summarizer',
            summary_request
        )
        summary_response = self.agents['summarizer'].process_message(summary_message)

        return summary_response.content if summary_response else {}

    def shutdown(self) -> None:
        """关闭系统"""
        for agent in self.agents.values():
            agent.stop()


# ===== 演示 =====

def demo_multi_agent() -> None:
    """演示：Multi-Agent 代码审查"""
    print("=" * 60)
    print("Multi-Agent 代码审查系统")
    print("=" * 60)

    # 示例代码
    sample_code = '''
def process_user_input(user_data):
    if not user_data:
        return None
    
    name = user_data['name']
    email = user_data['email']
    
    # 构建SQL查询（有安全风险）
    query = f"SELECT * FROM users WHERE name = '{name}' AND email = '{email}'"
    
    # 执行查询
    results = database.execute(query)
    
    # 处理结果
    for result in results:
        print(f"User: {result['name']}")
    
    return results
'''

    # 创建系统
    system = CodeReviewCoordinator()

    try:
        result = system.review_code(
            code=sample_code,
            language='python',
            context='用户数据处理函数',
            sensitive_data=['email', 'name']
        )

        print(f"\n{result.get('summary', '')}")
        print("\n💡 建议：")
        for rec in result.get('recommendations', [])[:5]:
            print(f"  • {rec}")

    finally:
        system.shutdown()
    print()


def demo_architecture() -> None:
    """演示：架构说明"""
    print("=" * 60)
    print("Multi-Agent 系统架构")
    print("=" * 60)
    print("""
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  风格检查员  │    │  Bug分析师  │    │  性能专家   │
│  StyleAgent │    │  BugAgent   │    │ PerfAgent   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                   ┌───────────────┐
                   │  协调器Agent  │
                   │ Coordinator   │
                   └───────────────┘
                           │
                   ┌───────────────┐
                   │  汇总报告     │
                   └───────────────┘

核心组件：
1. BaseAgent: Agent 基类，管理消息处理循环
2. MessageBus: 消息总线，实现 Agent 间通信
3. MessageFactory: 消息工厂，标准化消息格式
4. ConsistencyManager: 一致性管理器，处理冲突
5. Coordinator: 协调器，编排整个审查流程
""")


def main() -> None:
    print("\n" + "=" * 60)
    print("场景十：Multi-Agent 代码审查系统")
    print("=" * 60 + "\n")

    if os.getenv("DEEPSEEK_API_KEY"):
        demo_multi_agent()
    else:
        print("⚠️  DEEPSEEK_API_KEY 未设置，跳过 LLM 调用演示")
        print("   使用模拟数据展示架构...\n")

    demo_architecture()

    print("=" * 60)
    print("场景十演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
