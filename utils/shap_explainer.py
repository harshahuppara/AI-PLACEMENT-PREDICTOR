import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.model_loader import load_xgboost_model, load_feature_names

FEATURE_DISPLAY_NAMES = {
    "python_skill": "Python Skill Level",
    "java_skill": "Java Skill Level",
    "sql_skill": "SQL Skill Level",
    "react_skill": "React Skill Level",
    "aws_skill": "AWS Skill Level",
    "dsa_skill": "Data Structures & Algo",
    "interview_score": "Interview Performance Score",
    "aptitude_score": "General Aptitude Score",
    "quantitative_aptitude_score": "Quant Aptitude Score",
    "logical_aptitude_score": "Logical Reasoning Score",
    "communication_score": "Communication Skill Score",
    "verbal_fluency_score": "Verbal Fluency Score",
    "presentation_score": "Presentation Skill Score",
    "projects_count": "Projects Count",
    "project_complexity": "Project Complexity Level",
    "open_source_contributions": "Open Source Contributions",
    "hackathons_participated": "Hackathons Participated",
    "internships_count": "Internships Count",
    "cgpa": "Academic CGPA",
    "technical_skill_index": "Technical Skill Index",
    "aptitude_index": "Combined Aptitude Index",
    "communication_index": "Combined Communication Index",
    "practical_experience_index": "Practical Experience Index",
    "academic_index": "Academic Performance Index",
    "target_role_Full Stack Developer": "Target Role: Full Stack Dev",
    "target_role_Data Analyst": "Target Role: Data Analyst",
    "target_role_DevOps Engineer": "Target Role: DevOps Engineer",
    "target_role_QA Engineer": "Target Role: QA Engineer"
}

def get_shap_explanation(X_trans_instance):
    """
    Computes SHAP tree explanation for a single prediction instance.
    Returns positive factors, negative factors, and generates a clean waterfall-style matplotlib plot.
    """
    model = load_xgboost_model()
    feature_names = load_feature_names()

    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(X_trans_instance)
    shap_vals = shap_obj.values[0]

    # Map features and values
    feature_impacts = []
    for idx, f_name in enumerate(feature_names):
        val = shap_vals[idx]
        display_name = FEATURE_DISPLAY_NAMES.get(f_name, f_name.replace('_', ' ').title())
        feature_impacts.append({
            "feature": f_name,
            "display_name": display_name,
            "shap_value": float(val),
            "abs_shap": abs(float(val))
        })

    # Sort impacts
    sorted_impacts = sorted(feature_impacts, key=lambda x: x['abs_shap'], reverse=True)
    top_positives = [x for x in sorted_impacts if x['shap_value'] > 0][:6]
    top_negatives = [x for x in sorted_impacts if x['shap_value'] < 0][:6]

    return {
        "top_positives": top_positives,
        "top_negatives": top_negatives,
        "all_impacts": sorted_impacts,
        "shap_values": shap_vals,
        "feature_names": feature_names
    }

def render_shap_bar_chart(shap_explanation):
    """
    Generates a high-resolution horizontal bar chart of top positive and negative SHAP feature contributions.
    """
    positives = shap_explanation['top_positives'][:5]
    negatives = shap_explanation['top_negatives'][:5]
    
    combined = positives[::-1] + negatives
    if not combined:
        combined = shap_explanation['all_impacts'][:8]

    names = [item['display_name'] for item in combined]
    values = [item['shap_value'] for item in combined]
    colors = ['#10b981' if v > 0 else '#ef4444' for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0e1117')
    ax.set_facecolor('#1e222d')

    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor='none')
    ax.axvline(0, color='#6b7280', linestyle='--', linewidth=1)

    for bar, val in zip(bars, values):
        x_pos = val + (0.005 if val >= 0 else -0.015)
        align = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:+.3f}",
                va='center', ha=align, color='#ffffff', fontsize=9, fontweight='bold')

    ax.tick_params(colors='#ffffff', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')

    plt.title('SHAP Feature Impact on Placement Probability', color='#ffffff', fontsize=12, pad=15, fontweight='bold')
    plt.xlabel('SHAP Value (Contribution to Probability)', color='#9ca3af', fontsize=10)
    plt.tight_layout()
    
    return fig
