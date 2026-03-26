"""Agent 智能体"""
class Agent:
    def __init__(self, tools): self.tools = {t.name: t for t in tools}
    def run(self, msg): return "处理: " + msg
print(Agent([]).run("test"))
