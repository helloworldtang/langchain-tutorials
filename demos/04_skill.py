"""Skill 技能系统"""
class Skill:
    def __init__(self, name, prompt): self.name, self.prompt = name, prompt
skills = {"weather": Skill("weather", "天气助手")}
print(skills["weather"].name)
