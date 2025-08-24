from typing import Dict, List

class LearningPathAgent:
    def __init__(self, groq_agent):
        self.groq = groq_agent

    def suggest_learning_resources(self, skill: str) -> str:
        prompt = f"""Suggest the best learning resources for {skill} including:
        - Free and paid options
        - Different formats (videos, interactive, books)
        - Recommended platforms
        - Estimated time commitment
        - Prerequisites if any"""
        return self.groq.generate(prompt, [])

    def learning_roadmap(self, skill: str, level: str = "beginner") -> str:
        prompt = f"""Create a comprehensive learning roadmap for {skill} at {level} level with:
        - Phase-based structure (foundations, core concepts, advanced topics)
        - Recommended resources for each phase
        - Project ideas to practice
        - Time estimates for each phase
        - Milestones to track progress"""
        return self.groq.generate(prompt, [])

    def compare_learning_options(self, skill: str) -> str:
        prompt = f"""Compare different learning approaches for {skill}:
        - Self-paced vs structured courses
        - Free vs paid resources  
        - Video-based vs text-based vs interactive
        - Pros and cons of each approach
        - Recommendations based on learning style"""
        return self.groq.generate(prompt, [])

    def skill_prerequisites(self, target_skill: str) -> str:
        prompt = f"""What are the prerequisites for learning {target_skill}?
        - Foundational knowledge required
        - Related skills to learn first
        - Recommended learning order
        - Time investment for prerequisites"""
        return self.groq.generate(prompt, [])

    def project_ideas(self, skill: str, difficulty: str = "beginner") -> str:
        prompt = f"""Suggest practical project ideas for {skill} at {difficulty} level:
        - Project descriptions and goals
        - Skills practiced in each project
        - Estimated time to complete
        - Resources needed
        - Portfolio value"""
        return self.groq.generate(prompt, [])