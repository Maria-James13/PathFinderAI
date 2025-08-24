import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGAgent:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        
        # Load data
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(file_path)
        
        print(f"Loaded data with columns: {self.df.columns.tolist()}")
        print(f"Data shape: {self.df.shape}")
        
        # Clean data
        self._clean_data()
        self._create_combined_text()
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        print(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")

    def query(self, question: str, top_k=1) -> List[Dict[str, Any]]:
        """Query using TF-IDF cosine similarity - return only the most relevant result"""
        try:
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
            
            top_index = np.argmax(similarities)
            
            result = {
                'similarity': float(similarities[top_index]),
                'data': self.df.iloc[top_index].to_dict(),
                'text': self.df.iloc[top_index]['combined_text']
            }
            
            return [result] if result['similarity'] > 0.1 else []
            
        except Exception as e:
            print(f"Error in query: {e}")
            return []

    def _clean_data(self):
        """Clean and preprocess the learning resources data"""
        self.df = self.df.replace('', pd.NA)
        
        text_columns = ['Platform', 'Skill', 'Cost', 'Resource Type', 'Skill Level', 'Format']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Not specified').astype(str).str.strip()

    def _create_combined_text(self):
        """Combine all relevant columns into a single text field"""
        text_parts = []
        
        for _, row in self.df.iterrows():
            row_text = []
            for col, value in row.items():
                if col != 'combined_text' and pd.notna(value) and str(value).strip() != '':
                    row_text.append(f"{col}: {value}")
            combined = " | ".join(row_text)
            text_parts.append(combined)
        
        self.df['combined_text'] = text_parts

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning resources statistics"""
        stats = {} 
        
        stats['total_resources'] = len(self.df)
        stats['platforms_count'] = self.df['Platform'].nunique() if 'Platform' in self.df.columns else 0
        stats['skills_count'] = self.df['Skill'].nunique() if 'Skill' in self.df.columns else 0
        
        # Top Platforms
        if 'Platform' in self.df.columns:
            platform_counts = self.df['Platform'].value_counts().head(5)
            stats['top_platforms'] = [
                {'platform': platform, 'resources': count} 
                for platform, count in platform_counts.items() 
                if platform != 'Not specified'
            ]
        
        # Top Skills
        if 'Skill' in self.df.columns:
            skill_counts = self.df['Skill'].value_counts().head(5)
            stats['top_skills'] = [
                {'skill': skill, 'count': count} 
                for skill, count in skill_counts.items() 
                if skill != 'Not specified'
            ]
        
        # Cost Statistics 
        if 'Cost' in self.df.columns:
            cost_stats = self._analyze_costs()
            stats['cost'] = cost_stats
        
        # Resource Type Analysis 
        if 'Resource Type' in self.df.columns:
            type_counts = self.df['Resource Type'].value_counts()
            stats['resource_types'] = [
                {'type': rtype, 'count': count} 
                for rtype, count in type_counts.items() 
                if rtype != 'Not specified'
            ]
        
        # Availability Rate 
        available_resources = len(self.df[self.df['Platform'] != 'Not specified'])
        stats['availability_rate'] = (available_resources / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        return stats

    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost data for student understanding"""
        cost_data = self.df[self.df['Cost'] != 'Not specified']['Cost']
        numeric_values = []
        
        for cost in cost_data:
            if isinstance(cost, str):
                numbers = re.findall(r'(\d+\.?\d*)', cost.replace(',', ''))
                if numbers:
                    nums = [float(num) for num in numbers]
                    if len(nums) > 1 and any(char in cost for char in ['-', 'to']):
                        numeric_values.append(sum(nums) / len(nums))
                    else:
                        numeric_values.extend(nums)
        
        if numeric_values:
            return {
                'average': round(sum(numeric_values) / len(numeric_values), 2),
                'max': round(max(numeric_values), 2),
                'min': round(min(numeric_values), 2),
                'count': len(numeric_values),
                'common_range': self._get_common_range(numeric_values)
            }
        return {}

    def _get_common_range(self, values: List[float]) -> str:
        """Get the most common cost range"""
        if not values:
            return "Not available"
        
        ranges = {
            'Free': len([v for v in values if v == 0]),
            '$1-20': len([v for v in values if 0 < v <= 20]),
            '$20-50': len([v for v in values if 20 < v <= 50]),
            '$50-100': len([v for v in values if 50 < v <= 100]),
            '$100+': len([v for v in values if v > 100])
        }
        
        most_common = max(ranges.items(), key=lambda x: x[1])
        return most_common[0] if most_common[1] > 0 else "Not available"

    def analyze_data_with_groq(self, question: str, groq_agent) -> str:
        """Use Groq to analyze the data and answer complex questions"""
        try:
            data_summary = self._create_data_summary()
            
            prompt = f"""
            You are a learning resources analyst. Analyze the following learning resources data and answer the question.

            LEARNING RESOURCES DATA SUMMARY:
            {data_summary}

            QUESTION: {question}

            REQUIREMENTS:
            1. Analyze the actual data to provide accurate statistics
            2. Be specific and quantitative
            3. Include numbers and percentages where possible
            4. Provide insights based on the data
            5. If the question can't be answered from the data, explain why

            ANSWER:
            """
            
            response = groq_agent.generate(prompt, [], {})
            return response
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    def _create_data_summary(self) -> str:
        """Create a comprehensive summary of the data for Groq analysis"""
        summary = f"Total Learning Resources: {len(self.df)}\n\n"
        
        if 'Platform' in self.df.columns:
            platform_counts = self.df['Platform'].value_counts()
            summary += "TOP LEARNING PLATFORMS:\n"
            for platform, count in platform_counts.head(5).items():
                if platform != 'Not specified':
                    summary += f"- {platform}: {count} resources\n"
            summary += "\n"
        
        if 'Skill' in self.df.columns:
            skill_counts = self.df['Skill'].value_counts()
            summary += "TOP SKILLS COVERED:\n"
            for skill, count in skill_counts.head(5).items():
                if skill != 'Not specified':
                    summary += f"- {skill}: {count} resources\n"
            summary += "\n"
        
        if 'Cost' in self.df.columns:
            cost_stats = self._analyze_costs()
            summary += "COST STATS:\n"
            summary += f"- Average: ${cost_stats.get('average', 'N/A')}\n"
            summary += f"- Highest: ${cost_stats.get('max', 'N/A')}\n"
            summary += f"- Lowest: ${cost_stats.get('min', 'N/A')}\n"
            summary += "\n"
        
        if 'Skill Level' in self.df.columns:
            level_counts = self.df['Skill Level'].value_counts()
            summary += "SKILL LEVEL DISTRIBUTION:\n"
            for level, count in level_counts.items():
                summary += f"- {level}: {count} resources\n"
            summary += "\n"
        
        if 'Format' in self.df.columns:
            format_counts = self.df['Format'].value_counts()
            summary += "RESOURCE FORMAT DISTRIBUTION:\n"
            for fmt, count in format_counts.items():
                percentage = (count / len(self.df)) * 100
                summary += f"- {fmt}: {count} resources ({percentage:.1f}%)\n"
            summary += "\n"
        
        if 'Resource Type' in self.df.columns:
            type_counts = self.df['Resource Type'].value_counts()
            summary += "RESOURCE TYPES:\n"
            for rtype, count in type_counts.items():
                if rtype != 'Not specified':
                    percentage = (count / len(self.df)) * 100
                    summary += f"- {rtype}: {count} resources ({percentage:.1f}%)\n"
        
        return summary

    def get_skill_category_stats(self, skill_level: str) -> Dict[str, Any]:
        """Get statistics for specific skill level (e.g., Beginner, Advanced)"""
        if 'Skill Level' not in self.df.columns:
            return {}
        
        level_data = self.df[self.df['Skill Level'].str.contains(skill_level, case=False, na=False)]
        
        if len(level_data) == 0:
            return {}
        
        stats = {
            'total_resources': len(level_data),
            'availability_rate': 0,
            'top_platforms': [],
            'top_skills': [],
            'average_cost': 0
        }
        
        available_resources = level_data[level_data['Platform'] != 'Not specified']
        stats['availability_rate'] = (len(available_resources) / len(level_data)) * 100 if len(level_data) > 0 else 0
        
        if 'Platform' in level_data.columns:
            platform_counts = level_data['Platform'].value_counts().head(3)
            stats['top_platforms'] = [{'platform': p, 'count': c} for p, c in platform_counts.items() if p != 'Not specified']
        
        if 'Skill' in level_data.columns:
            skill_counts = level_data['Skill'].value_counts().head(3)
            stats['top_skills'] = [{'skill': s, 'count': c} for s, c in skill_counts.items() if s != 'Not specified']
        
        if 'Cost' in level_data.columns:
            cost_values = []
            for cost in level_data['Cost']:
                if cost != 'Not specified':
                    numbers = re.findall(r'(\d+\.?\d*)', str(cost).replace(',', ''))
                    if numbers:
                        nums = [float(num) for num in numbers]
                        cost_values.append(sum(nums) / len(nums))
            
            if cost_values:
                stats['average_cost'] = sum(cost_values) / len(cost_values)
        
        return stats

    def get_columns(self) -> List[str]:
        return self.df.columns.tolist()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_rows': len(self.df),
            'columns': self.get_columns(),
            'platforms': self.df['Platform'].dropna().unique().tolist() if 'Platform' in self.df.columns else [],
            'skills': self.df['Skill'].dropna().unique().tolist() if 'Skill' in self.df.columns else []
        }

    def search_by_platform(self, platform_name: str) -> List[Dict]:
        """Search by learning platform"""
        if 'Platform' not in self.df.columns:
            return []
        
        results = self.df[self.df['Platform'].str.contains(platform_name, case=False, na=False)]
        return results.to_dict('records')

    def search_by_skill(self, skill_name: str) -> List[Dict]:
        """Search by skill"""
        if 'Skill' not in self.df.columns:
            return []
        
        results = self.df[self.df['Skill'].str.contains(skill_name, case=False, na=False)]
        return results.to_dict('records')
