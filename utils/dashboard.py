import pandas as pd
import numpy as np
from utils.model_loader import load_dataset
from utils.prediction import predict_placement

def get_institutional_data(filtered_df=None):
    """
    Computes institutional placement metrics across dataset.
    Returns overall summary statistics and dataframes for charts.
    """
    df = load_dataset() if filtered_df is None else filtered_df

    if len(df) == 0:
        return {}, pd.DataFrame()

    total_students = len(df)
    placed_count = (df['placement_outcome'] == 'Placed').sum()
    not_placed_count = (df['placement_outcome'] == 'Not Placed').sum()

    ready_count = (df['placement_status'] == 'Ready').sum()
    near_ready_count = (df['placement_status'] == 'Near Ready').sum()
    needs_training_count = (df['placement_status'] == 'Needs Training').sum()

    avg_prob = df['placement_probability'].mean()
    avg_cgpa = df['cgpa'].mean()
    avg_aptitude = df['aptitude_score'].mean()
    avg_interview = df['interview_score'].mean()
    avg_comm = df['communication_score'].mean()
    avg_projects = df['projects_count'].mean()
    avg_hackathons = df['hackathons_participated'].mean()

    summary = {
        "total_students": total_students,
        "placed_count": placed_count,
        "not_placed_count": not_placed_count,
        "placed_pct": round((placed_count / total_students) * 100.0, 1),
        "ready_count": ready_count,
        "near_ready_count": near_ready_count,
        "needs_training_count": needs_training_count,
        "avg_probability": round(avg_prob, 1),
        "avg_cgpa": round(avg_cgpa, 2),
        "avg_aptitude": round(avg_aptitude, 1),
        "avg_interview": round(avg_interview, 1),
        "avg_communication": round(avg_comm, 1),
        "avg_projects": round(avg_projects, 1),
        "avg_hackathons": round(avg_hackathons, 1)
    }

    # Department breakdown
    dept_stats = df.groupby('department').agg(
        total_students=('student_id', 'count'),
        ready_count=('placement_status', lambda x: (x == 'Ready').sum()),
        near_ready_count=('placement_status', lambda x: (x == 'Near Ready').sum()),
        needs_training_count=('placement_status', lambda x: (x == 'Needs Training').sum()),
        placed_count=('placement_outcome', lambda x: (x == 'Placed').sum()),
        not_placed_count=('placement_outcome', lambda x: (x == 'Not Placed').sum()),
        avg_probability=('placement_probability', 'mean'),
        avg_cgpa=('cgpa', 'mean'),
        avg_aptitude=('aptitude_score', 'mean'),
        avg_interview=('interview_score', 'mean')
    ).reset_index()

    dept_stats['avg_probability'] = dept_stats['avg_probability'].round(1)
    dept_stats['avg_cgpa'] = dept_stats['avg_cgpa'].round(2)
    dept_stats['avg_aptitude'] = dept_stats['avg_aptitude'].round(1)
    dept_stats['avg_interview'] = dept_stats['avg_interview'].round(1)

    return summary, dept_stats

def get_vulnerable_students(df, threshold_prob=60.0):
    """
    Identifies students with low placement probability (< threshold_prob) or 'Needs Training' status.
    """
    vulnerable = df[
        (df['placement_probability'] < threshold_prob) | 
        (df['placement_status'] == 'Needs Training')
    ].copy()

    # Determine primary weak area
    def get_weak_area(row):
        scores = {
            "Aptitude Score": row['aptitude_score'],
            "Interview Score": row['interview_score'],
            "Communication": row['communication_score'],
            "Coding Performance": row['coding_score'],
            "Projects": row['projects_count'] * 15,
            "CGPA": row['cgpa'] * 10
        }
        return min(scores, key=scores.get)

    vulnerable['primary_weak_area'] = vulnerable.apply(get_weak_area, axis=1)

    display_cols = [
        'student_id', 'student_name', 'department', 'cgpa',
        'placement_probability', 'placement_status', 'target_role', 'primary_weak_area'
    ]
    return vulnerable[display_cols].sort_values('placement_probability', ascending=True)

def get_skill_heatmap_data(df):
    """
    Generates department-level skill averages matrix for heatmap display.
    """
    skills = [
        'python_skill', 'java_skill', 'sql_skill', 'react_skill', 'aws_skill', 'dsa_skill',
        'aptitude_score', 'communication_score', 'interview_score', 'projects_count', 'hackathons_participated'
    ]

    rename_map = {
        'python_skill': 'Python',
        'java_skill': 'Java',
        'sql_skill': 'SQL',
        'react_skill': 'React',
        'aws_skill': 'AWS',
        'dsa_skill': 'DSA',
        'aptitude_score': 'Aptitude',
        'communication_score': 'Communication',
        'interview_score': 'Interview',
        'projects_count': 'Projects',
        'hackathons_participated': 'Hackathons'
    }

    heatmap_df = df.groupby('department')[skills].mean().rename(columns=rename_map)
    
    # Normalize ratings (0-3 vs 0-100) to 0-100 scale for intuitive heatmap comparison
    heatmap_df['Python'] = (heatmap_df['Python'] / 3.0 * 100).round(1)
    heatmap_df['Java'] = (heatmap_df['Java'] / 3.0 * 100).round(1)
    heatmap_df['SQL'] = (heatmap_df['SQL'] / 3.0 * 100).round(1)
    heatmap_df['React'] = (heatmap_df['React'] / 3.0 * 100).round(1)
    heatmap_df['AWS'] = (heatmap_df['AWS'] / 3.0 * 100).round(1)
    heatmap_df['DSA'] = (heatmap_df['DSA'] / 3.0 * 100).round(1)
    heatmap_df['Projects'] = (heatmap_df['Projects'] / 8.0 * 100).round(1)
    heatmap_df['Hackathons'] = (heatmap_df['Hackathons'] / 5.0 * 100).round(1)

    return heatmap_df.round(1)
