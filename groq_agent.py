
from groq import Groq
from typing import List, Dict
import os

class GroqAgent:
    def __init__(self, model="llama-3.1-8b-instant"):
        api_key = ""
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, context: List[Dict] = None, conversation_topics: Dict = None) -> str:
        try:
            # Build system message with context guidance
            system_message = {
                "role": "system",
                "content": f"""You are a career placement assistant. Maintain conversation context and provide continuous, contextual responses.

IMPORTANT: Remember the entire conversation history and continue naturally from previous messages.

Current conversation topics: {conversation_topics if conversation_topics else 'New conversation'}

Guidelines:
1. Continue the conversation naturally from previous context
2. Reference previous questions and answers when relevant
3. Maintain topic continuity
4. If user changes topic, acknowledge but try to connect to previous context
5. Be helpful and provide specific information about careers, placements, skills, and companies"""
            }
            
            # Prepare messages
            messages = [system_message]
            if context:
                messages.extend(context)
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I apologize, I'm having trouble responding right now. Error: {str(e)}"