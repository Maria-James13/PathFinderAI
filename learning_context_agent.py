from typing import Dict, List
import re

class LearningContextAgent:
    def __init__(self, max_context_length=12):
        self.context: Dict[str, List[Dict]] = {}
        self.max_context_length = max_context_length

    def update_context(self, session_id: str, role: str, content: str):
        if session_id not in self.context:
            self.context[session_id] = []
        
        self.context[session_id].append({"role": role, "content": content})
        
        if len(self.context[session_id]) > self.max_context_length:
            self.context[session_id] = self.context[session_id][-self.max_context_length:]

    def get_context(self, session_id: str) -> List[Dict]:
        return self.context.get(session_id, [])

    def clear_context(self, session_id: str):
        if session_id in self.context:
            self.context[session_id] = []

    def get_conversation_summary(self, session_id: str) -> str:
        """Create a summary of the learning conversation for context continuity"""
        context = self.get_context(session_id)
        if not context:
            return "No previous conversation about learning paths."
        
        summary = "Learning Conversation History:\n"
        for msg in context:
            summary += f"{msg['role']}: {msg['content']}\n"
        
        return summary

    def extract_conversation_topics(self, session_id: str) -> Dict[str, List[str]]:
        """Extract learning topics, platforms, and skills from conversation"""
        context = self.get_context(session_id)
        topics = {
            'skills': [],
            'platforms': [],
            'learning_levels': [],
            'resource_types': [],
            'general_topics': []
        }
        
        for msg in context:
            content = msg['content'].lower()
            
            # Extract skills
            skill_keywords = [
                'python', 'javascript', 'java', 'sql', 'react', 'node', 'angular', 
                'vue', 'typescript', 'html', 'css', 'data analysis', 'data science',
                'machine learning', 'deep learning', 'ai', 'artificial intelligence',
                'web development', 'mobile development', 'cloud computing', 'aws',
                'azure', 'google cloud', 'devops', 'cybersecurity', 'ui/ux design',
                'digital marketing', 'project management', 'blockchain', 'game development'
            ]
            for skill in skill_keywords:
                if skill in content:
                    topics['skills'].append(skill)
            
            # Extract learning platforms
            platform_keywords = [
                'coursera', 'udemy', 'edx', 'youtube', 'pluralsight', 'linkedin learning',
                'skillshare', 'freecodecamp', 'khan academy', 'codecademy', 'udacity',
                'futurelearn', 'codewars', 'leetcode', 'hackerrank', 'datacamp'
            ]
            for platform in platform_keywords:
                if platform in content:
                    topics['platforms'].append(platform)
            
            # Extract learning levels
            level_keywords = ['beginner', 'intermediate', 'advanced', 'all levels', 'expert']
            for level in level_keywords:
                if level in content:
                    topics['learning_levels'].append(level)
            
            # Extract resource types
            resource_keywords = ['free', 'paid', 'subscription', 'course', 'tutorial', 
                               'book', 'video', 'interactive', 'project-based', 'bootcamp']
            for resource in resource_keywords:
                if resource in content:
                    topics['resource_types'].append(resource)
            
            # Extract general learning topics
            general_keywords = ['learn', 'study', 'practice', 'master', 'understand', 
                              'skill', 'career', 'job', 'placement', 'interview',
                              'portfolio', 'certificate', 'certification']
            for keyword in general_keywords:
                if keyword in content:
                    topics['general_topics'].append(keyword)
        
        # Remove duplicates
        for key in topics:
            topics[key] = list(set(topics[key]))
        
        return topics

    def get_learning_goals(self, session_id: str) -> List[str]:
        """Extract learning goals from conversation context"""
        context = self.get_context(session_id)
        goals = []
        
        goal_patterns = [
            r'want to learn (.*?)',
            r'want to master (.*?)',
            r'want to become (.*?)',
            r'need to learn (.*?)',
            r'study (.*?)',
            r'improve my (.*?)',
            r'build skills in (.*?)'
        ]
        
        for msg in context:
            content = msg['content'].lower()
            for pattern in goal_patterns:
                matches = re.findall(pattern, content)
                goals.extend(matches)
        
        return list(set(goals))

    def get_preferred_learning_style(self, session_id: str) -> Dict[str, bool]:
        """Detect preferred learning style from conversation"""
        context = self.get_context(session_id)
        learning_style = {
            'video': False,
            'interactive': False,
            'text': False,
            'project_based': False,
            'self_paced': False
        }
        
        for msg in context:
            content = msg['content'].lower()
            
            if any(word in content for word in ['video', 'watch', 'youtube']):
                learning_style['video'] = True
            if any(word in content for word in ['interactive', 'practice', 'hands-on', 'code along']):
                learning_style['interactive'] = True
            if any(word in content for word in ['read', 'book', 'article', 'text']):
                learning_style['text'] = True
            if any(word in content for word in ['project', 'build', 'create', 'portfolio']):
                learning_style['project_based'] = True
            if any(word in content for word in ['self-paced', 'at my own pace', 'flexible']):
                learning_style['self_paced'] = True
        
        return learning_style