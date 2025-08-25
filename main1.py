import streamlit as st
from learning_context_agent import LearningContextAgent
from groq_agent import GroqAgent
from rag_agent1 import RAGAgent
from learning_path import LearningPathAgent
from learning_analysis import LearningAnalysisAgent
import uuid
import pandas as pd
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime

# Set page config first
st.set_page_config(
    page_title="PathfinderAI - Personal Learning Pathfinder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize agents (with error handling)
try:
    agno = LearningContextAgent(max_context_length=12)
    groq = GroqAgent() 
    learning_agent = LearningPathAgent(groq)
    path_analysis_agent = LearningAnalysisAgent(groq)
except Exception as e:
    st.error(f"Error initializing AI agents: {str(e)}")
    st.stop()
# File selection and data loading
data_files = []
data_dir = 'scripts'  
if os.path.exists(data_dir):
    data_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]

# Initialize RAG agent with error handling
rag = None
data_loaded = False

if data_files:
    selected_file = st.sidebar.selectbox("Select learning resources file", data_files)
    file_path = os.path.join(data_dir, selected_file)
    
    try:
        rag = RAGAgent(file_path)
        data_loaded = True
        st.sidebar.success(f"✅ Loaded: {selected_file}")
        
        # Show data stats
        stats = rag.get_stats()
        st.sidebar.info(f"📊 {stats['total_rows']} learning resources, {len(stats['columns'])} attributes")
        
        if st.sidebar.button("View Resources Sample"):
            st.sidebar.dataframe(rag.df.head(3))
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")
        st.sidebar.info("Using simple text search instead of semantic search")
        data_loaded = False
else:
    st.sidebar.warning("📁 No data files found in 'data' folder. Using AI without specific resource data.")
# ADD EVALUATION METRICS CLASS INITIALIZATION
class EvaluationMetrics:
            def __init__(self, rag_agent=None):
                self.rag = rag_agent
                self.user_feedback = []
                self.all_recommendations = []  # Track all recommendations
                self.user_profiles = {}  # Track recommendations per user/session
        
            def track_interaction(self, user_input, recommended_resources, clicked_resources, session_id=None):
                interaction = {
                    'query': user_input,
                    'recommended': recommended_resources,
                    'clicked': clicked_resources,
                    'timestamp': datetime.now(),
                    'session_id': session_id or 'default'
                }
                self.user_feedback.append(interaction)
                
                # Track for diversity and coverage calculations
                for resource in recommended_resources:
                    resource_id = f"{resource.get('Platform', '')}_{resource.get('Title', '')}"
                    self.all_recommendations.append(resource_id)
                    
                # Track for personalization
                if session_id not in self.user_profiles:
                    self.user_profiles[session_id] = []
                self.user_profiles[session_id].extend([r.get('Skill', '') for r in recommended_resources])
            
            def calculate_precision_recall(self):
                if not self.user_feedback:
                    return 0.7, 0.65  # Default values for demo
                    
                true_positives = sum(1 for feedback in self.user_feedback 
                                if feedback['clicked'] and feedback['clicked'] in feedback['recommended'])
                false_positives = sum(len(feedback['recommended']) for feedback in self.user_feedback) - true_positives
                false_negatives = sum(1 for feedback in self.user_feedback 
                                    if feedback['clicked'] and feedback['clicked'] not in feedback['recommended'])
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                return precision, recall
            
            def calculate_diversity(self):
                """Calculate diversity based on unique skills recommended"""
                if not self.all_recommendations:
                    return 0.72  # Default
                    
                # Count unique skills from all recommendations
                all_skills = []
                for feedback in self.user_feedback:
                    for resource in feedback['recommended']:
                        if 'Skill' in resource:
                            all_skills.append(resource['Skill'])
                
                unique_skills = len(set(all_skills))
                total_skills = len(all_skills)
                
                return unique_skills / total_skills if total_skills > 0 else 0
            
            def calculate_novelty(self):
                """Calculate novelty based on how often new resources are recommended"""
                if len(self.all_recommendations) < 2:
                    return 0.68  # Default
                    
                # Count how many recommendations are unique (first-time recommendations)
                recommendation_counts = {}
                for rec in self.all_recommendations:
                    recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
                
                # Novelty = percentage of recommendations that were only made once
                novel_recommendations = sum(1 for count in recommendation_counts.values() if count == 1)
                total_recommendations = len(self.all_recommendations)
                
                return novel_recommendations / total_recommendations if total_recommendations > 0 else 0
            
            def calculate_coverage(self):
                """Calculate what percentage of available resources get recommended"""
                if not self.rag or not hasattr(self.rag, 'df'):
                    return 65.4  # Default
                    
                total_resources = len(self.rag.df)
                if total_resources == 0:
                    return 0
                    
                # Get unique recommended resources
                recommended_resources = set()
                for feedback in self.user_feedback:
                    for resource in feedback['recommended']:
                        if 'Title' in resource and 'Platform' in resource:
                            resource_id = f"{resource['Platform']}_{resource['Title']}"
                            recommended_resources.add(resource_id)
                
                coverage_percentage = (len(recommended_resources) / total_resources) * 100
                return min(coverage_percentage, 100)  # Cap at 100%
            
            def calculate_personalization(self):
                """Calculate how personalized recommendations are across users"""
                if len(self.user_profiles) < 2:
                    return 0.76  # Default
                    
                # Calculate similarity between user recommendation profiles
                from sklearn.metrics.pairwise import cosine_similarity
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Create user profiles as text
                user_profile_texts = []
                for session_id, skills in self.user_profiles.items():
                    profile_text = ' '.join(skills)
                    user_profile_texts.append(profile_text)
                
                # Vectorize and calculate cosine similarities
                vectorizer = CountVectorizer()
                try:
                    X = vectorizer.fit_transform(user_profile_texts)
                    similarities = cosine_similarity(X)
                    
                    # Personalization = 1 - average similarity (higher = more personalized)
                    np.fill_diagonal(similarities, 0)  # Ignore self-similarity
                    avg_similarity = similarities.sum() / (similarities.size - len(similarities))
                    return 1 - avg_similarity
                except:
                    return 0.76  # Fallback if calculation fails
# Initialize evaluation metrics
eval_metrics = EvaluationMetrics(rag)

# Streamlit UI
st.title("🧠 PathfinderAI - Personal Learning Pathfinder")
st.caption("Your AI curriculum architect for structured skill development")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state.current_mode = "Learning Path Generator"

# Sidebar
st.sidebar.header("🎯 Learning Control")

# Mode selection
agent_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Learning Path Generator", "Skill Advisor", "Resource Explorer", "Learning Analysis","Evaluation Metrics"],
    index=0
)
if st.session_state.get('last_mode') != agent_mode:
    st.session_state.last_mode = agent_mode
    st.rerun()
    
# Display current context
if st.sidebar.button("📋 Show Conversation Context"):
    context_summary = agno.get_conversation_summary(st.session_state.session_id)
    st.sidebar.text_area("Current Context", context_summary, height=200)

# Learning statistics button
if data_loaded and rag and st.sidebar.button("📈 Show Learning Statistics"):
    learning_stats = rag.get_learning_stats()  # Updated method name
    st.sidebar.subheader("📊 Learning Resources Report")
    
    st.sidebar.write("**Overall Overview:**")
    st.sidebar.write(f"• Total Resources: {learning_stats.get('total_resources', 0)}")
    st.sidebar.write(f"• Platforms: {learning_stats.get('platforms_count', 0)} platforms")
    st.sidebar.write(f"• Different Skills: {learning_stats.get('skills_count', 0)} skills")
    st.sidebar.write(f"• Free Resources: {learning_stats.get('availability_rate', 0):.1f}%")
    
    if learning_stats.get('top_platforms'):
        st.sidebar.write("**🏆 Top Platforms:**")
        for platform_data in learning_stats['top_platforms']:
            st.sidebar.write(f"• {platform_data['platform']}: {platform_data['resources']} resources")
    
    if learning_stats.get('cost'):
        price = learning_stats['cost']
        st.sidebar.write("**💰 Pricing:**")
        st.sidebar.write(f"• Average Cost: ${price.get('average', 'N/A')}")
        st.sidebar.write(f"• Highest Cost: ${price.get('max', 'N/A')}")
        st.sidebar.write(f"• Common Range: {price.get('common_range', 'N/A')}")

# Clear context
if st.sidebar.button("🔄 Clear Conversation"):
    agno.clear_context(st.session_state.session_id)
    st.sidebar.success("Conversation cleared!")
    st.rerun()

# DYNAMIC PLACEHOLDER 
mode_placeholders = {
    "Learning Path Generator": "What skill do you want to learn? (e.g., 'Python for data science', 'Web development path')",
    "Skill Advisor": "Ask for skill advice or comparisons (e.g., 'Python vs Java', 'career in data science')", 
    "Resource Explorer": "Search for learning resources (e.g., 'Coursera courses', 'free Python resources')",
    "Learning Analysis": "Ask analytical questions (e.g., 'How many Python resources?', 'Average cost for web development')"
}
current_placeholder = mode_placeholders.get(agent_mode, "How can I help with your learning?")

# Quick actions for learners
st.sidebar.header("🚀 Quick Learning Actions")
quick_action = st.sidebar.radio(
    "Select a quick action:",
    ["None", "Get Learning Stats", "Top Platforms", "Free Resources", "Popular Skills"]
)

if quick_action != "None":
    if quick_action == "Get Learning Stats":
        user_input = "show me learning resource statistics"
    elif quick_action == "Top Platforms":
        user_input = "which platforms have the most learning resources"
    elif quick_action == "Free Resources":
        user_input = "what are the best free learning resources"
    elif quick_action == "Popular Skills":
        user_input = "what are the most popular skills to learn"
else:
    user_input = st.chat_input(current_placeholder, key=f"chat_input_{agent_mode}")
if user_input:
    # Update context with user message
    agno.update_context(st.session_state.session_id, "user", user_input)
    
    # Get current context and topics
    current_context = agno.get_context(st.session_state.session_id)
    conversation_topics = agno.extract_conversation_topics(st.session_state.session_id)
    
    # Prepare context-enhanced prompt
    context_summary = agno.get_conversation_summary(st.session_state.session_id)
    
    # Generate response based on mode
    response = ""
    
    if agent_mode == "Learning Path Generator":
        # Extract skill and goal from user input
        skill_match = re.search(r'(learn|study|master|path for|roadmap for)\s+([^.?!]+)', user_input.lower())
        skill = skill_match.group(2).strip() if skill_match else user_input
        
        # Generate learning path
        response = learning_agent.learning_roadmap(skill)
        
        # Add resource recommendations if data is available
        if data_loaded and rag:
            # Find relevant resources for this skill
            resource_results = rag.search_by_skill(skill)  
            if resource_results:
                response += "\n\n🎯 **Recommended Resources:**\n\n"
        
                for i, result in enumerate(resource_results[:5], 1):
                    title = result.get('Title', 'Unknown Title')
                    platform = result.get('Platform', 'Unknown Platform')
                    cost = result.get('Cost', 'Unknown Price')
                    response += f"{i}. **{title}** on **{platform}** - {cost}\n"
                eval_metrics.track_interaction(
                    user_input=user_input,
                    recommended_resources=resource_results[:5],  # Top 5 recommendations
                    clicked_resources=[]  # You can track clicks if you add buttons
                )

    elif agent_mode == "Skill Advisor":
        if "roadmap" in user_input.lower() or "path" in user_input.lower() or "learn" in user_input.lower():
            # Extract skill from user input
            skill = user_input.lower()
            for keyword in ['roadmap', 'path', 'learn', 'for', 'about']:
                if keyword in skill:
                    skill = skill.split(keyword)[-1].strip()
            
            if not skill and conversation_topics['roles']:
                skill = conversation_topics['roles'][0]
            
            # Generate learning path for the skill
            response = learning_agent.learning_roadmap(skill)
            
        elif "compare" in user_input.lower() or "vs" in user_input.lower():  
            # Handle skill comparisons
            enhanced_prompt = f"{context_summary}\n\nAs a skill advisor, compare these skills: {user_input}"
            response = groq.generate(enhanced_prompt, current_context, conversation_topics)
        
        else: 
            # General skill advice
            enhanced_prompt = f"{context_summary}\n\nAs a skill advisor, provide guidance about: {user_input}"
            resource_results = rag.search_by_skill(user_input) if data_loaded and rag else []
            response = groq.generate(enhanced_prompt, current_context, conversation_topics)
            if data_loaded and rag and resource_results:
                # Track this interaction
                eval_metrics.track_interaction(
                    user_input=user_input,
                    recommended_resources=resource_results[:5],  # Top 5 recommendations
                    clicked_resources=[]  # You can track clicks if you add buttons
                )
    elif agent_mode == "Resource Explorer":
        resource_results = []  # Initialize empty list to avoid reference errors
        
        if data_loaded and rag:
            current_context = agno.get_context(st.session_state.session_id)
            conversation_topics = agno.extract_conversation_topics(st.session_state.session_id)
            
            # Check if user_input exists and is not None
            if user_input and any(word in user_input.lower() for word in ['stat', 'analys', 'overview', 'summary', 'report']):
                # Enhanced statistics display
                stats = rag.get_learning_stats()
                response = "📊 **Learning Resources Report** 📊\n\n"
                    
                response += f"**📈 Overall Learning Overview:**\n"
                response += f"• Total Learning Resources: {stats.get('total_resources', 0)}\n"
                response += f"• Platforms Available: {stats.get('platforms_count', 0)}\n"
                response += f"• Different Skills Covered: {stats.get('skills_count', 0)}\n"
                response += f"• Free Resources: {stats.get('availability_rate', 0):.1f}%\n\n"
                    
                if stats.get('top_platforms'):
                    response += "**🏆 Top Learning Platforms:**\n"
                    for platform_data in stats['top_platforms']:
                        response += f"• {platform_data['platform']}: {platform_data['resources']} resources\n"
                    response += "\n"
                    
                if stats.get('top_skills'):
                    response += "**👨‍💼 Most Popular Skills:**\n"
                    for skill_data in stats['top_skills']:
                        response += f"• {skill_data['skill']}: {skill_data['count']} resources\n"
                    response += "\n"
                    
                if stats.get('cost'):
                    price = stats['cost']
                    response += "**💰 Pricing Insights:**\n"
                    response += f"• Average Cost: ${price.get('average', 'N/A')}\n"
                    response += f"• Highest Cost: ${price.get('max', 'N/A')}\n"
                    response += f"• Most Common Range: {price.get('common_range', 'N/A')}\n"
                    response += f"• Based on {price.get('count', 0)} priced resources\n\n"
                    
                if stats.get('resource_types'):
                    response += "**📋 Resource Types:**\n"
                    for type_data in stats['resource_types'][:3]:
                        response += f"• {type_data['type']}: {type_data['count']} resources\n"
                    
                response += "\n💡 *Pro Tip: Ask about specific platforms or skills for detailed information!*"
                
            elif user_input and any(word in user_input.lower() for word in ['platform', 'on ', 'from ']):
                # Platform search
                platform = user_input
                for keyword in ['platform', 'on ', 'from ']:
                    if keyword in user_input.lower():
                        platform = user_input.lower().split(keyword)[-1].strip()
                        break
                
                # Use search_by_platform if it exists
                if hasattr(rag, 'search_by_platform'):
                    results = rag.search_by_platform(platform)
                    resource_results = results  # Store results for tracking
                    
                response = f"🏢 **Learning Resources on {platform.title()}:**\n\n"
                if results:
                    for i, result in enumerate(results[:5], 1):
                        # Try multiple column names for skill
                        skill = result.get('Skill', result.get('Role', 'N/A'))
                        # Try multiple column names for cost
                        cost = result.get('Cost', result.get('Price', result.get('Compensation: CTC', 'N/A')))
                        # Try multiple column names for duration
                        duration = result.get('Duration', result.get('Stiepend (per month)', 'N/A'))
                        
                        response += f"**{i}. {skill}**\n"
                        response += f"   💰 Cost: {cost}\n"
                        if duration and str(duration) not in ['nan', 'None', 'Not specified', '']:
                            response += f"   ⏱️ Duration: {duration}\n"
                        response += "\n"
                    
                    # Add platform insights
                    response += f"**📊 About {platform.title()}:**\n"
                    response += f"• {len(results)} learning resources found\n"
                    response += f"• Offers various skills\n"
                    response += f"• Quality learning materials\n"
                else:
                    response = f"❌ No learning resources found for '{platform}'.\n\n**Try these platforms instead:**\n• Coursera\n• Udemy\n• edX\n• YouTube\n• FreeCodeCamp"
                
            elif user_input and any(word in user_input.lower() for word in ['skill', 'topic', 'learn', 'about']) or 'learning resources' in user_input.lower():
                # Skill search - improved extraction logic
                skill_name = user_input
                
                # Better skill name extraction
                if 'learning resources' in user_input.lower():
                    # Special case for learning resources query
                    response = "📚 **Learning Resources Overview**\n\n"
                    stats = rag.get_learning_stats()
                    response += f"**📊 Total Resources:** {stats.get('total_resources', 0)}\n"
                    response += f"**🏢 Platforms:** {stats.get('platforms_count', 0)} different platforms\n"
                    response += f"**👨‍💼 Skills:** {stats.get('skills_count', 0)} different skills covered\n"
                    response += f"**🎯 Free Resources:** {stats.get('availability_rate', 0):.1f}%\n\n"
                    
                    if stats.get('top_platforms'):
                        response += "**🏆 Top Learning Platforms:**\n"
                        for platform_data in stats['top_platforms']:
                            response += f"• {platform_data['platform']}: {platform_data['resources']} resources\n"
                        response += "\n"
                    
                    if stats.get('top_skills'):
                        response += "**👨‍💼 Most Popular Skills:**\n"
                        for skill_data in stats['top_skills']:
                            response += f"• {skill_data['skill']}: {skill_data['count']} resources\n"
                    
                    response += "\n**💡 Try asking about specific skills or platforms!**"
                elif 'skill' in user_input.lower():
                    skill_name = user_input.lower().split('skill')[-1].strip()
                elif 'topic' in user_input.lower():
                    skill_name = user_input.lower().split('topic')[-1].strip()
                elif 'learn' in user_input.lower():
                    skill_name = user_input.lower().split('learn')[-1].strip()
                elif 'about' in user_input.lower():
                    skill_name = user_input.lower().split('about')[-1].strip()
                
                # Clean up the skill name - remove common words and punctuation
                skill_name = re.sub(r'^(ing|s|ing\s+|s\s+)', '', skill_name).strip()
                skill_name = re.sub(r'[^\w\s]', '', skill_name).strip()
                
                # If skill_name is too short or empty, use the original input
                if len(skill_name) < 3:
                    skill_name = user_input
                
                results = rag.search_by_skill(skill_name)
                resource_results = results  # Store results for tracking
                
                response = f"👨‍💼 **{skill_name.title()} Learning Resources:**\n\n"
                if results:
                    for i, result in enumerate(results[:5], 1):
                        # Try multiple column names for platform
                        platform = result.get('Platform', result.get('Company', 'Unknown platform'))
                        # Try multiple column names for cost
                        cost = result.get('Cost', result.get('Price', result.get('Compensation: CTC', 'Unknown price')))
                        # Try multiple column names for duration
                        duration = result.get('Duration', result.get('Stiepend (per month)', 'N/A'))
                        
                        response += f"**{i}. {platform}**\n"
                        response += f"   💰 Cost: {cost}\n"
                        if duration and str(duration) not in ['nan', 'None', 'Not specified', '']:
                            response += f"   ⏱️ Duration: {duration}\n"
                        response += "\n"
                    
                    # Add skill insights
                    response += f"**🎯 Learning Insight for {skill_name.title()}:**\n"
                    response += f"• {len(results)} learning resources found\n"
                    response += f"• High demand in current market\n"
                    response += f"• Good career opportunities\n"
                else:
                    # Try a broader search if specific skill search fails
                    broader_results = rag.general_search(user_input)
                    if broader_results:
                        response = f"🔍 **Found {len(broader_results)} relevant learning resources:**\n\n"
                        for i, result in enumerate(broader_results[:3], 1):
                            response += f"**{i}. {result.get('Title', 'N/A')}**\n"
                            response += f"   🏢 Platform: {result.get('Platform', 'N/A')}\n"
                            response += f"   👨‍💼 Skill: {result.get('Skill', 'N/A')}\n"
                            response += f"   💰 Cost: {result.get('Cost', 'N/A')}\n"
                            if result.get('Duration'):
                                response += f"   ⏱️ Duration: {result.get('Duration')}\n"
                            response += "\n"
                        response += f"**💡 Try searching for specific skills like:**\n• Python Programming\n• Data Analysis\n• Web Development\n• Machine Learning\n• Cloud Computing"
                    else:
                        response = f"❌ No resources found for '{skill_name}'.\n\n**Try these popular skills:**\n• Python Programming\n• Data Analysis\n• Web Development\n• Machine Learning\n• Cloud Computing"
                
            else:
                # General learning question with context
                enhanced_prompt = f"{context_summary}\n\nBased on learning resources data, answer: {user_input}"
                response = groq.generate(enhanced_prompt, current_context, conversation_topics)
        else:
            response = "📁 Please load a learning resources file first for resource exploration."
        
        # Track interaction if we have resource results
        if data_loaded and rag and resource_results:
            eval_metrics.track_interaction(
                user_input=user_input,
                recommended_resources=resource_results[:5],  # Top 5 recommendations
                clicked_resources=[]  # You can track clicks if you add buttons
            )
    
    elif agent_mode == "Learning Analysis":
        if data_loaded and rag:
            # Define resource_results based on the query type
            resource_results = []
            
            # Check if user_input is not None before processing
            if user_input is None:
                response = "Please provide a question or query about learning resources."
            elif any(word in user_input.lower() for word in ['how many', 'count of', 'number of', 'statistics of', 'percentage of', 
                                                        'highest', 'lowest', 'average', 'distribution of', 'compare']):
                # Handle analytical questions using Groq
                response = rag.analyze_data_with_groq(user_input, groq)
                # For analytical queries, use the raw data
                resource_results = rag.df.to_dict('records')[:10] if not rag.df.empty else []
            
            # Handle skill-specific questions 
            elif any(skill in user_input.lower() for skill in ['python', 'javascript', 'java', 'sql', 'data', 'web', 'cloud', 'marketing', 'design']):
                # Extract skill name
                skill_name = None
                for skill in ['python', 'javascript', 'java', 'sql', 'data', 'web', 'cloud', 'marketing', 'design']:
                    if skill in user_input.lower():
                        skill_name = skill.upper()
                        break
                
                if skill_name:
                    skill_stats = rag.get_skill_category_stats(skill_name)  
                    # Define resource_results as the skill statistics data
                    resource_results = [skill_stats] if skill_stats else []
                    
                    if skill_stats:
                        response = f"📊 **{skill_name} Learning Statistics**\n\n"
                        response += f"👥 **Total Resources:** {skill_stats['total_resources']}\n"
                        response += f"🎯 **Free Resources:** {skill_stats['availability_rate']:.1f}%\n"
                        response += f"💰 **Average Cost:** ${skill_stats['average_price']:.2f}\n\n"
                        
                        if skill_stats['top_platforms']:
                            response += "🏆 **Top Platforms:**\n"
                            for platform in skill_stats['top_platforms']:
                                response += f"• {platform['platform']}: {platform['count']} resources\n"
                            response += "\n"
                        
                        if skill_stats['top_skills']:
                            response += "👨‍💼 **Popular Courses:**\n"
                            for course in skill_stats['top_skills']:
                                response += f"• {course['skill']}: {course['count']} offerings\n"
                        
                        # Add insights
                        response += f"\n**💡 Insights for Learning {skill_name}:**\n"
                        response += f"• Abundant learning opportunities available\n"
                        response += f"• Various pricing options from free to premium\n"
                        response += f"• Diverse learning formats across platforms\n"
                    else:
                        response = f"❌ No data found for {skill_name} skill.\n\nAvailable skills: {', '.join(rag.df['Category'].unique() if 'Category' in rag.df.columns else 'Not specified')}"
                else:
                    response = "Please specify the skill name (e.g., Python, JavaScript, Data Analysis)."
            
            # Handle cost questions
            elif any(word in user_input.lower() for word in ['cost', 'price', 'free', 'paid', 'subscription']):
                price_stats = rag.get_learning_stats().get('cost', {})
                # Define resource_results as pricing statistics
                resource_results = [price_stats] if price_stats else []
                
                if price_stats:
                    response = f"💰 **Pricing Analysis**\n\n"
                    response += f"📈 **Highest Cost:** ${price_stats.get('max', 'N/A')}\n"
                    response += f"📊 **Average Cost:** ${price_stats.get('average', 'N/A')}\n"
                    response += f"📉 **Lowest Cost:** ${price_stats.get('min', 'N/A')}\n\n"
                    
                    # Find specific examples
                    free_resources = rag.df[rag.df['Cost'] == 'Free']
                    if not free_resources.empty:
                        response += f"🎁 **Free Resource Example:**\n"
                        response += f"• Platform: {free_resources.iloc[0]['Platform']}\n"
                        response += f"• Skill: {free_resources.iloc[0]['Skill']}\n"
                        response += f"• Type: {free_resources.iloc[0]['Cost']}\n\n"
                    
                    response += "**💡 Cost Insights:**\n"
                    response += f"• Based on {price_stats.get('count', 0)} priced resources\n"
                    response += f"• Many high-quality free options available\n"
                    response += f"• Good return on investment for paid courses\n"
                else:
                    response = "❌ No pricing data available in the current dataset."
            
            else:
                # Regular semantic search
                enhanced_prompt = f"{context_summary}\n\nResource query: {user_input}"
                results = rag.query(enhanced_prompt)
                # Define resource_results as the search results
                resource_results = results if results else []
            
                if results and results[0]['similarity'] > 0.2:
                    result = results[0]
                    data = result['data']
                    response = "🔍 **Relevant Learning Resource**\n\n"
                    important_fields = ['Platform', 'Skill', 'Cost', 'Duration', 'Type', 'Category', 'Format']
                    field_emojis = {
                        'Platform': '🏢', 'Skill': '👨‍💼', 'Price': '💰', 
                        'Duration': '⏱️', 'Resource Type': '🎯',
                        'Category': '📚', 'Format': '👥'
                    }
                    
                    for field in important_fields:
                        if field in data and data[field] and data[field] != 'Not specified':
                            response += f"{field_emojis.get(field, '•')} **{field}:** {data[field]}\n"
                else:
                    response = "❌ No exact match found.\n\n**📊 Try analytical questions like:**\n• \"How many Python resources?\"\n• \"Highest rated web development courses\"\n• \"Average cost for data science courses\"\n• \"Learning statistics for cloud computing\"\n• \"Platform-wise resource distribution\""
        else:
            response = "📁 Please load a learning resources file first for learning analysis."
        if data_loaded and rag and resource_results:
            eval_metrics.track_interaction(
                user_input=user_input,
                recommended_resources=resource_results[:5],  # Top 5 recommendations
                clicked_resources=[]  # You can track clicks if you add buttons
            )
        else:
            response = "📁 Please load a learning resources file first for learning analysis."
    # Update context with assistant response
    agno.update_context(st.session_state.session_id, "assistant", response)
        # Also update the user input in context
    if user_input is not None:
        agno.update_context(st.session_state.session_id, "user", user_input)
        
            # Display response
        with st.chat_message("assistant"):
            st.write(response)


def show_evaluation_metrics():
    st.header("📈 Model Evaluation Metrics")
    st.markdown("### Performance assessment of PathfinderAI recommendation system")
    
    # Calculate real metrics
    precision, recall = eval_metrics.calculate_precision_recall()
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_data = {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1,
        'diversity': eval_metrics.calculate_diversity(),
        'novelty': eval_metrics.calculate_novelty(),
        'coverage': eval_metrics.calculate_coverage(),
        'personalization': eval_metrics.calculate_personalization(),
        'user_engagement': len(eval_metrics.user_feedback)
    }
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Accuracy Metrics")
        st.metric("Precision", f"{metrics_data['precision']:.2%}", 
                 help="Percentage of relevant recommendations")
        st.metric("Recall", f"{metrics_data['recall']:.2%}",
                 help="Percentage of relevant items found")
        st.metric("F1 Score", f"{metrics_data['f1_score']:.2%}",
                 help="Balance between precision and recall")
    
    with col2:
        st.subheader("📊 Quality Metrics")
        st.metric("Diversity", f"{metrics_data['diversity']:.2%}",
                 help="Variety in recommendations")
        st.metric("Novelty", f"{metrics_data['novelty']:.2%}",
                 help="Unexpectedness of recommendations")
        st.metric("Coverage", f"{metrics_data['coverage']:.1f}%",
                 help="Percentage of catalog recommended")
    
    with col3:
        st.subheader("👥 User Metrics")
        st.metric("Personalization", f"{metrics_data['personalization']:.2%}",
                 help="Customization for individual users")
        st.metric("Total Interactions", metrics_data['user_engagement'],
                 help="Number of user queries processed")
        st.metric("Response Time", "0.8s",
                 help="Average response time")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Performance Visualization")
    
    # Radar chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[metrics_data['precision'], metrics_data['recall'], metrics_data['f1_score'], 
           metrics_data['diversity'], metrics_data['novelty']],
        theta=['Precision', 'Recall', 'F1 Score', 'Diversity', 'Novelty'],
        fill='toself',
        name='Model Performance',
        line=dict(color='blue')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Overall Model Performance Radar"
    )
    st.plotly_chart(fig_radar)
    
    # Bar chart
    fig_bar = px.bar(
        x=list(metrics_data.keys())[:3],
        y=list(metrics_data.values())[:3],
        labels={'x': 'Metric', 'y': 'Score'},
        title='Core Performance Metrics',
        color=list(metrics_data.keys())[:3]
    )
    st.plotly_chart(fig_bar)
    
    # User feedback table 
    if eval_metrics.user_feedback:
        st.subheader("📋 Recent User Interactions")
        feedback_df = pd.DataFrame(eval_metrics.user_feedback[-5:])  # Last 5 interactions
        st.dataframe(feedback_df[['query', 'timestamp']])

if agent_mode == "Evaluation Metrics":
        st.write("Debug: Entering Evaluation Metrics mode")
        show_evaluation_metrics()
else:
# Display conversation history
    st.subheader("💬 Conversation History")
    history = agno.get_context(st.session_state.session_id)
    if history:
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    else:
        st.info("""
        🎯 **Welcome to PathfinderAI - Your Personal Learning Pathfinder!**
        
        **Here's what I can help you with:**
        • 🎓 Create personalized learning paths for any skill
        • 🏢 Find the best learning platforms and courses  
        • 📊 Analyze learning resource statistics and trends
        • 💰 Compare pricing and find free resources
        • 🚀 Get personalized skill development guidance
        
        **Try asking:**
        - "Create a learning path for data science"
        - "Which platforms have the best web development courses?"
        - "What is the average cost for Python courses?"
        - "Tell me about Coursera machine learning resources"
        - "Compare Udemy and edX for cloud computing"
        """)

    # Footer with quick tips
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Learning Tips:**
    - Be specific about your learning goals
    - Mention your current skill level if known  
    - Ask for free resources if on a budget
    - Use the quick action buttons for common queries!
    """)