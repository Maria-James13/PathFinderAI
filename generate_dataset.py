import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

def generate_learning_resources_dataset(num_records=1000, output_path="../scripts/learning_resources.csv"):
    """
    Generate a synthetic dataset of learning resources for the Learning Path Assistant.
    
    Args:
        num_records (int): Number of records to generate
        output_path (str): Path to save the CSV file
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    # Initialize faker for realistic data generation
    fake = Faker()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define possible values for categorical fields
    platforms = ['Coursera', 'Udemy', 'edX', 'YouTube', 'Pluralsight', 'LinkedIn Learning', 
                 'Skillshare', 'FreeCodeCamp', 'Khan Academy', 'Codecademy', 'Udacity', 'FutureLearn']
    
    skills = ['Python Programming', 'Data Analysis', 'Web Development', 'Machine Learning', 
              'Cloud Computing', 'JavaScript', 'SQL', 'React', 'DevOps', 'UI/UX Design',
              'Mobile Development', 'Data Science', 'Cybersecurity', 'Digital Marketing',
              'Project Management', 'Artificial Intelligence', 'Blockchain', 'Game Development']
    
    levels = ['Beginner', 'Intermediate', 'Advanced', 'All Levels']
    formats = ['Video Course', 'Interactive', 'Text-based', 'Project-based', 'Bootcamp', 'Tutorial Series']
    resource_types = ['Free', 'Paid', 'Subscription', 'Freemium']
    duration_units = ['hours', 'days', 'weeks', 'months']
    
    ratings = [round(np.random.uniform(3.0, 5.0), 1) for _ in range(num_records)]
    
    # Generate synthetic data
    data = []
    for i in range(num_records):
        platform = random.choice(platforms)
        skill = random.choice(skills)
        level = random.choice(levels)
        format_type = random.choice(formats)
        resource_type = random.choice(resource_types)
        
        # Generate realistic titles based on platform and skill
        if platform == 'Coursera':
            title = f"{skill} Specialization"
        elif platform == 'Udemy':
            title = f"The Complete {skill} Bootcamp"
        elif platform == 'edX':
            title = f"{skill} Professional Certificate"
        elif platform == 'YouTube':
            title = f"Learn {skill} - Full Course"
        else:
            title = f"{skill} Masterclass"
        
        # Generate realistic URLs
        base_url = platform.lower().replace(' ', '')
        course_slug = title.lower().replace(' ', '-')
        url = f"https://www.{base_url}.com/courses/{course_slug}"
        
        # Generate realistic pricing based on platform and resource type
        if resource_type == 'Free':
            cost = 0
        elif resource_type == 'Paid':
            cost = round(random.uniform(10, 200), 2)
        elif resource_type == 'Subscription':
            cost = round(random.uniform(15, 50), 2)
        else:  # Freemium
            cost = round(random.uniform(0, 100), 2)
        
        # Generate duration
        duration_value = random.randint(1, 30)
        duration_unit = random.choice(duration_units)
        duration = f"{duration_value} {duration_unit}"
        
        # Generate enrollment count (more popular for some platforms)
        if platform in ['Coursera', 'Udemy', 'YouTube']:
            enrollments = random.randint(1000, 50000)
        else:
            enrollments = random.randint(100, 10000)
        
        # Generate release date (within last 3 years)
        days_ago = random.randint(1, 1095)  # 3 years
        release_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Generate last updated date (after release date)
        update_days = random.randint(0, days_ago)
        last_updated = (datetime.now() - timedelta(days=update_days)).strftime('%Y-%m-%d')
        
        # Generate instructor (sometimes platform as instructor)
        if random.random() < 0.7:
            instructor = fake.name()
        else:
            instructor = platform
        
        # Generate certificate availability
        has_certificate = random.random() < 0.6
        
        # Generate hands-on projects availability
        has_projects = random.random() < 0.7
        
        # Generate community support availability
        has_community = random.random() < 0.5
        
        # Generate rating count (correlated with enrollments)
        rating_count = int(enrollments * random.uniform(0.1, 0.3))
        
        # Generate difficulty score (1-10)
        if level == 'Beginner':
            difficulty = random.randint(1, 4)
        elif level == 'Intermediate':
            difficulty = random.randint(4, 7)
        elif level == 'Advanced':
            difficulty = random.randint(7, 10)
        else:  # All Levels
            difficulty = random.randint(3, 8)
        
        # Generate completion rate (percentage)
        completion_rate = random.randint(30, 95)
        
        # Generate prerequisites
        prerequisites = []
        if level != 'Beginner':
            num_prereqs = random.randint(1, 3)
            for _ in range(num_prereqs):
                prereq_skill = random.choice(skills)
                if prereq_skill != skill:
                    prerequisites.append(prereq_skill)
        
        # Generate related skills
        related_skills = []
        num_related = random.randint(1, 4)
        for _ in range(num_related):
            related_skill = random.choice(skills)
            if related_skill != skill and related_skill not in prerequisites:
                related_skills.append(related_skill)
        
        # Create the record
        record = {
            'Resource_ID': f"RES_{i:04d}",
            'Title': title,
            'Platform': platform,
            'Skill': skill,
            'Level': level,
            'Format': format_type,
            'Type': resource_type,
            'Cost': cost,
            'URL': url,
            'Duration': duration,
            'Enrollments': enrollments,
            'Rating': ratings[i],
            'Rating_Count': rating_count,
            'Release_Date': release_date,
            'Last_Updated': last_updated,
            'Instructor': instructor,
            'Has_Certificate': has_certificate,
            'Has_Projects': has_projects,
            'Has_Community': has_community,
            'Difficulty_Score': difficulty,
            'Completion_Rate': completion_rate,
            'Prerequisites': ', '.join(prerequisites) if prerequisites else 'None',
            'Related_Skills': ', '.join(related_skills)
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    # Generate the dataset when this script is run directly
    print("Generating synthetic learning resources dataset...")
    df = generate_learning_resources_dataset(num_records=1000)
    print(f"Dataset created with {len(df)} records!")
    print("Saved to ../data/learning_resources.csv")
    print("\nDataset columns:", df.columns.tolist())
    print("\nFirst 3 records:")
    print(df.head(3))