import os
import pandas as pd
import numpy as np
import streamlit as st
from utils.model_loader import load_xgboost_model, load_preprocessor, load_feature_names

DATASET_PATH = "AI_Placement_Predictor_Dataset_V3.csv"

EXCLUDED_COLUMNS = [
    'student_id',
    'student_name',
    'placement_outcome',
    'placement_probability',
    'placement_status',
    'missing_skills',
    'recommended_courses',
    'roadmap_weeks'
]

def engineer_features(data):
    """
    Applies exact row-level feature engineering without dropping original raw features.
    Matches the training feature pipeline.
    """
    df_feat = data.copy()

    # Parse pipe-separated programming languages into binary indicators & count
    languages_list = ['Python', 'Java', 'C++']
    for lang in languages_list:
        col_name = f"lang_{lang.lower().replace('++', 'cpp')}"
        df_feat[col_name] = df_feat['programming_languages'].apply(
            lambda x: 1 if pd.notnull(x) and lang in str(x).split('|') else 0
        )
    df_feat['num_prog_languages'] = df_feat['programming_languages'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) and str(x) != 'None' else 0
    )

    # Parse pipe-separated frameworks
    frameworks_list = ['React', 'Spring Boot', 'Pandas', 'AWS', 'Git']
    for fw in frameworks_list:
        col_name = f"fw_{fw.lower().replace(' ', '_')}"
        df_feat[col_name] = df_feat['frameworks'].apply(
            lambda x: 1 if pd.notnull(x) and fw in str(x).split('|') else 0
        )
    df_feat['num_frameworks'] = df_feat['frameworks'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) and str(x) != 'None' else 0
    )

    # Parse pipe-separated verified certifications
    cert_list = ['Python Certificate', 'SQL Certificate', 'Cloud Certificate', 'DSA Certificate', 'Web Development Certificate']
    for cert in cert_list:
        col_name = f"cert_{cert.lower().split()[0]}"
        df_feat[col_name] = df_feat['verified_certifications'].apply(
            lambda x: 1 if pd.notnull(x) and cert in str(x).split('|') else 0
        )
    df_feat['num_verified_certs'] = df_feat['verified_certifications'].apply(
        lambda x: 0 if pd.isnull(x) or str(x) == 'None' else len(str(x).split('|'))
    )

    # Composite Domain Indices
    tech_skill_cols = ['python_skill', 'java_skill', 'sql_skill', 'react_skill', 'aws_skill', 'dsa_skill']
    df_feat['technical_skill_index'] = df_feat[tech_skill_cols].mean(axis=1) + (df_feat['num_prog_languages'] * 0.5) + (df_feat['num_frameworks'] * 0.5)

    df_feat['practical_experience_index'] = (
        df_feat['projects_count'] * 1.5 + 
        df_feat['project_complexity'] * 2.0 + 
        df_feat['internships_count'] * 3.0 + 
        (df_feat['open_source_contributions'] * 0.5)
    )

    df_feat['aptitude_index'] = (
        df_feat['aptitude_score'] + 
        df_feat['quantitative_aptitude_score'] + 
        df_feat['logical_aptitude_score']
    ) / 3.0

    df_feat['communication_index'] = (
        df_feat['communication_score'] + 
        df_feat['verbal_fluency_score'] + 
        df_feat['interview_score'] + 
        df_feat['presentation_score']
    ) / 4.0

    df_feat['extracurricular_index'] = (
        df_feat['hackathons_participated'] * 1.0 + 
        df_feat['leadership_roles'] * 1.5 + 
        df_feat['tech_society_participation'] * 1.0
    )

    df_feat['academic_index'] = (
        (df_feat['cgpa'] * 10.0) * 0.5 + 
        df_feat['twelfth_percentage'] * 0.25 + 
        df_feat['tenth_percentage'] * 0.25 - 
        (df_feat['backlogs'] * 5.0)
    )

    df_feat['overall_readiness_index'] = (
        df_feat['academic_index'] * 0.25 +
        df_feat['technical_skill_index'] * 10.0 * 0.25 +
        df_feat['aptitude_index'] * 0.25 +
        df_feat['communication_index'] * 0.25
    )

    df_feat = df_feat.drop(columns=['programming_languages', 'frameworks', 'verified_certifications'], errors='ignore')
    return df_feat


def predict_placement(student_data):
    """
    Executes inference for student data dict or DataFrame using trained XGBoost model.
    """
    if isinstance(student_data, dict):
        df_input = pd.DataFrame([student_data])
    elif isinstance(student_data, pd.DataFrame):
        df_input = student_data.copy()
    else:
        raise TypeError("student_data must be a dict or pandas DataFrame")

    if 'verified_certifications' in df_input.columns:
        df_input['verified_certifications'] = df_input['verified_certifications'].fillna('None')

    df_eng = engineer_features(df_input)
    preprocessor = load_preprocessor()
    model = load_xgboost_model()

    input_cols = [c for c in df_eng.columns if c not in EXCLUDED_COLUMNS]
    X_trans = preprocessor.transform(df_eng[input_cols])

    probs = model.predict_proba(X_trans)[:, 1]
    preds = (probs >= 0.50).astype(int)

    results = []
    for i in range(len(df_input)):
        prob_pct = float(probs[i] * 100.0)
        prediction_label = "Placed" if preds[i] == 1 else "Not Placed"
        
        # Readiness derivation
        if prob_pct >= 80.0:
            readiness = "Ready"
            badge_color = "#10b981"
        elif prob_pct >= 50.0:
            readiness = "Near Ready"
            badge_color = "#f59e0b"
        else:
            readiness = "Needs Training"
            badge_color = "#ef4444"

        confidence_pct = float(probs[i] if preds[i] == 1 else (1.0 - probs[i])) * 100.0

        results.append({
            "placement_probability": round(prob_pct, 1),
            "placement_prediction": prediction_label,
            "readiness_status": readiness,
            "badge_color": badge_color,
            "confidence": round(confidence_pct, 1),
            "raw_probability": float(probs[i]),
            "X_trans_instance": X_trans[i:i+1]
        })

    return results if len(results) > 1 else results[0]


def save_student_to_dataset(student_data, pred_res, gaps=None, roadmap=None):
    """
    Saves student employability profile to AI_Placement_Predictor_Dataset_V3.csv.
    - Ignores re-entered duplicate names within the SAME department.
    - If new student name in department, assigns auto-incremented student_id starting from STU1001+.
    - Instantly updates dataset and clears Streamlit cache for live TPO Dashboard display.
    """
    name = str(student_data.get('student_name', 'Student')).strip()
    dept = str(student_data.get('department', '')).strip()

    if not name or name.lower() in ["applicant student", "new student", "student"]:
        return False, "Please enter a valid student name.", None

    if not os.path.exists(DATASET_PATH):
        return False, f"Dataset file {DATASET_PATH} not found.", None

    df = pd.read_csv(DATASET_PATH)

    # Duplicate check: Same student name in the SAME department
    existing_dup = df[
        (df['student_name'].astype(str).str.strip().str.lower() == name.lower()) & 
        (df['department'].astype(str).str.strip().str.lower() == dept.lower())
    ]

    if len(existing_dup) > 0:
        existing_id = existing_dup.iloc[0]['student_id']
        return False, f"Student '{name}' is already registered in '{dept}' department ({existing_id}). Duplicate entry not saved to dataset.", existing_id

    # Auto-increment student_id starting from STU1001+
    stu_nums = []
    for sid in df['student_id'].astype(str):
        if sid.startswith("STU"):
            try:
                stu_nums.append(int(sid.replace("STU", "")))
            except ValueError:
                pass

    max_num = max(stu_nums) if stu_nums else 1000
    next_id = f"STU{max_num + 1:04d}"

    # Extract missing skills
    missing_str = "None"
    if gaps:
        missing = [g['skill'] for g in gaps if g.get('gap', 0) > 0]
        if missing:
            missing_str = "|".join(missing)

    # Extract recommended courses
    rec_str = "None"
    if roadmap and 'phases' in roadmap:
        rec_courses = []
        for phase in roadmap['phases']:
            rec_courses.extend(phase.get('recommended_courses', []))
        if rec_courses:
            rec_str = "|".join(list(dict.fromkeys(rec_courses)))

    weeks = roadmap.get('total_estimated_weeks', 8) if roadmap else 8

    new_row = {
        "student_id": next_id,
        "student_name": name,
        "department": dept,
        "semester": int(student_data.get("semester", 7)),
        "cgpa": float(student_data.get("cgpa", 7.5)),
        "tenth_percentage": float(student_data.get("tenth_percentage", 75.0)),
        "twelfth_percentage": float(student_data.get("twelfth_percentage", 75.0)),
        "backlogs": int(student_data.get("backlogs", 0)),
        "programming_languages": str(student_data.get("programming_languages", "Python")),
        "frameworks": str(student_data.get("frameworks", "React")),
        "verified_certifications": str(student_data.get("verified_certifications", "None")),
        "python_skill": int(student_data.get("python_skill", 0)),
        "java_skill": int(student_data.get("java_skill", 0)),
        "sql_skill": int(student_data.get("sql_skill", 0)),
        "react_skill": int(student_data.get("react_skill", 0)),
        "aws_skill": int(student_data.get("aws_skill", 0)),
        "dsa_skill": int(student_data.get("dsa_skill", 0)),
        "certifications_count": int(student_data.get("certifications_count", 0)),
        "projects_count": int(student_data.get("projects_count", 0)),
        "project_complexity": int(student_data.get("project_complexity", 1)),
        "open_source_contributions": int(student_data.get("open_source_contributions", 0)),
        "internships_count": int(student_data.get("internships_count", 0)),
        "aptitude_score": int(student_data.get("aptitude_score", 50)),
        "quantitative_aptitude_score": int(student_data.get("quantitative_aptitude_score", 50)),
        "logical_aptitude_score": int(student_data.get("logical_aptitude_score", 50)),
        "coding_score": int(student_data.get("coding_score", 50)),
        "communication_score": int(student_data.get("communication_score", 50)),
        "verbal_fluency_score": int(student_data.get("verbal_fluency_score", 50)),
        "interview_score": int(student_data.get("interview_score", 50)),
        "presentation_score": int(student_data.get("presentation_score", 50)),
        "hackathons_participated": int(student_data.get("hackathons_participated", 0)),
        "leadership_roles": int(student_data.get("leadership_roles", 0)),
        "tech_society_participation": int(student_data.get("tech_society_participation", 0)),
        "target_role": str(student_data.get("target_role", "Full Stack Developer")),
        "placement_probability": float(pred_res.get("placement_probability", 50.0)),
        "placement_status": str(pred_res.get("readiness_status", "Near Ready")),
        "missing_skills": missing_str,
        "recommended_courses": rec_str,
        "roadmap_weeks": int(weeks),
        "placement_outcome": str(pred_res.get("placement_prediction", "Not Placed"))
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    try:
        df.to_csv(DATASET_PATH, index=False)
    except (PermissionError, OSError) as pe:
        return False, f"Could not write to {DATASET_PATH} (File is locked/open in another program). Prediction generated successfully.", None

    try:
        st.cache_data.clear()
    except Exception:
        pass

    return True, f"Successfully stored student record as {next_id} for {name}! Now live in TPO Dashboard.", next_id
