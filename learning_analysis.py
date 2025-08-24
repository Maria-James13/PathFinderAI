from typing import Dict, List

class LearningAnalysisAgent:
    def __init__(self, groq_agent):
        self.groq = groq_agent

    def analyze_learning_resource(self, resource_name: str, platform: str = "") -> str:
        prompt = f"""Analyze the learning resource '{resource_name}' {f'on {platform} ' if platform else ''}:
        - Quality assessment and rating prediction
        - Target audience and difficulty level
        - Time commitment required
        - Value for money (if paid)
        - Pros and cons
        - Comparison to alternative resources
        - Recommended learning path placement"""
        return self.groq.generate(prompt, [])

    def skill_market_analysis(self, skill: str) -> str:
        prompt = f"""Provide a market analysis for {skill} skills:
        - Current demand in job market
        - Salary ranges and career opportunities
        - Future outlook and trends
        - Related skills that complement it
        - Industries and companies hiring for this skill
        - Learning investment vs career return"""
        return self.groq.generate(prompt, [])

    def learning_style_assessment(self, user_description: str) -> str:
        prompt = f"""Based on this learning style description: '{user_description}'
        Analyze the optimal learning approach:
        - Recommended resource types (video, text, interactive)
        - Best platforms for this learning style
        - Study techniques and strategies
        - Time management suggestions
        - Potential challenges and solutions"""
        return self.groq.generate(prompt, [])

    def resource_comparison(self, resource1: str, resource2: str) -> str:
        prompt = f"""Compare these two learning resources:
        Resource 1: {resource1}
        Resource 2: {resource2}
        
        Provide comparison on:
        - Content quality and depth
        - Teaching style and engagement
        - Time commitment
        - Cost effectiveness
        - Target audience suitability
        - Practical application value
        - Overall recommendation"""
        return self.groq.generate(prompt, [])

    def learning_progress_assessment(self, current_skills: str, goal: str) -> str:
        prompt = f"""Assess learning progress and provide guidance:
        Current skills: {current_skills}
        Learning goal: {goal}
        
        Provide:
        - Gap analysis between current and target skills
        - Recommended next steps and resources
        - Timeline estimation
        - Potential obstacles and solutions
        - Milestones to track progress
        - Motivation and study tips"""
        return self.groq.generate(prompt, [])