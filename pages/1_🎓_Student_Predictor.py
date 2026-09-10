import streamlit as st
import pandas as pd
import numpy as np
import importlib
import utils.prediction
importlib.reload(utils.prediction)
from utils.prediction import predict_placement, save_student_to_dataset
from utils.shap_explainer import get_shap_explanation, render_shap_bar_chart
from utils.skill_gap import analyze_skill_gap
from utils.roadmap import generate_personalized_roadmap

st.set_page_config(page_title="Student Placement Predictor | SIH", page_icon="🎓", layout="wide")

st.title("🎓 Student Placement Predictor & Employability Diagnostics")
st.caption("Complete 52-Feature Student Profile Evaluation powered by trained XGBoost Classification Engine & SHAP TreeExplainer")

st.markdown("---")

# Demo Profiles Presets Dictionary
DEMO_PRESETS = {
    "Select a Demo Profile to Auto-Fill Form...": None,
    "🌟 Profile 1: High Performing Candidate (Ready)": {
        "student_name": "Rohan Sharma",
        "cgpa": 9.2, "tenth_pct": 92.0, "twelfth_pct": 88.0, "backlogs": 0, "semester": 8, "department": "CSE",
        "python_skill": 3, "java_skill": 3, "sql_skill": 3, "react_skill": 3, "aws_skill": 2, "dsa_skill": 3,
        "prog_langs": ["Python", "Java", "C++"], "frameworks": ["React", "Spring Boot", "Pandas", "AWS"],
        "verified_certs": ["Python Certificate", "SQL Certificate", "DSA Certificate"],
        "projects_count": 5, "project_complexity": 5, "open_source": 15, "internships": 2,
        "aptitude_score": 90, "quant_score": 92, "logical_score": 88, "coding_score": 88,
        "comm_score": 92, "verbal_score": 90, "interview_score": 94, "presentation_score": 92,
        "hackathons": 4, "leadership": 2, "tech_society": 2, "target_role": "Full Stack Developer"
    },
    "📈 Profile 2: Balanced Candidate (Near Ready)": {
        "student_name": "Priya Verma",
        "cgpa": 7.5, "tenth_pct": 78.0, "twelfth_pct": 75.0, "backlogs": 0, "semester": 7, "department": "IT",
        "python_skill": 2, "java_skill": 2, "sql_skill": 2, "react_skill": 1, "aws_skill": 1, "dsa_skill": 2,
        "prog_langs": ["Python", "Java"], "frameworks": ["React", "Pandas"],
        "verified_certs": ["Python Certificate", "SQL Certificate"],
        "projects_count": 3, "project_complexity": 3, "open_source": 6, "internships": 1,
        "aptitude_score": 72, "quant_score": 74, "logical_score": 70, "coding_score": 68,
        "comm_score": 80, "verbal_score": 78, "interview_score": 76, "presentation_score": 78,
        "hackathons": 2, "leadership": 1, "tech_society": 1, "target_role": "Full Stack Developer"
    },
    "⚠️ Profile 3: Candidate Needing Training (Needs Training)": {
        "student_name": "Amit Kumar",
        "cgpa": 5.5, "tenth_pct": 58.0, "twelfth_pct": 55.0, "backlogs": 3, "semester": 6, "department": "Mechanical",
        "python_skill": 1, "java_skill": 0, "sql_skill": 1, "react_skill": 0, "aws_skill": 0, "dsa_skill": 0,
        "prog_langs": ["Python"], "frameworks": [],
        "verified_certs": [],
        "projects_count": 1, "project_complexity": 1, "open_source": 1, "internships": 0,
        "aptitude_score": 48, "quant_score": 45, "logical_score": 50, "coding_score": 42,
        "comm_score": 60, "verbal_score": 58, "interview_score": 52, "presentation_score": 55,
        "hackathons": 0, "leadership": 0, "tech_society": 0, "target_role": "QA Engineer"
    },
    "💻 Profile 4: Tech Specialist / Low Soft Skills": {
        "student_name": "Siddharth Tech",
        "cgpa": 8.1, "tenth_pct": 84.0, "twelfth_pct": 82.0, "backlogs": 0, "semester": 8, "department": "CSE",
        "python_skill": 3, "java_skill": 3, "sql_skill": 3, "react_skill": 3, "aws_skill": 3, "dsa_skill": 3,
        "prog_langs": ["Python", "Java", "C++"], "frameworks": ["React", "Spring Boot", "Pandas", "AWS", "Git"],
        "verified_certs": ["Python Certificate", "SQL Certificate", "DSA Certificate"],
        "projects_count": 6, "project_complexity": 5, "open_source": 18, "internships": 2,
        "aptitude_score": 86, "quant_score": 88, "logical_score": 84, "coding_score": 90,
        "comm_score": 52, "verbal_score": 50, "interview_score": 54, "presentation_score": 50,
        "hackathons": 5, "leadership": 0, "tech_society": 1, "target_role": "DevOps Engineer"
    },
    "🗣️ Profile 5: Soft Skills Specialist / Low Tech": {
        "student_name": "Ananya Voice",
        "cgpa": 7.6, "tenth_pct": 79.0, "twelfth_pct": 77.0, "backlogs": 0, "semester": 7, "department": "ECE",
        "python_skill": 1, "java_skill": 0, "sql_skill": 1, "react_skill": 0, "aws_skill": 0, "dsa_skill": 0,
        "prog_langs": ["Python"], "frameworks": ["Pandas"],
        "verified_certs": [],
        "projects_count": 1, "project_complexity": 2, "open_source": 2, "internships": 1,
        "aptitude_score": 68, "quant_score": 65, "logical_score": 70, "coding_score": 50,
        "comm_score": 95, "verbal_score": 96, "interview_score": 92, "presentation_score": 95,
        "hackathons": 1, "leadership": 3, "tech_society": 3, "target_role": "QA Engineer"
    }
}

# Quick Load Demo Selector
demo_col1, demo_col2 = st.columns([2, 1])
with demo_col1:
    selected_preset_name = st.selectbox("⚡ Quick Load Demo Preset (Optional)", list(DEMO_PRESETS.keys()))

preset = DEMO_PRESETS.get(selected_preset_name)

# Form Defaults
d_name = preset["student_name"] if preset else "Aarav Gupta"
d_cgpa = preset["cgpa"] if preset else 7.50
d_tenth = preset["tenth_pct"] if preset else 78.0
d_twelfth = preset["twelfth_pct"] if preset else 75.0
d_backlogs = preset["backlogs"] if preset else 0
d_sem = preset["semester"] if preset else 7
d_dept = preset["department"] if preset else "CSE"

d_py = preset["python_skill"] if preset else 2
d_java = preset["java_skill"] if preset else 1
d_sql = preset["sql_skill"] if preset else 2
d_react = preset["react_skill"] if preset else 1
d_aws = preset["aws_skill"] if preset else 1
d_dsa = preset["dsa_skill"] if preset else 2

d_langs = preset["prog_langs"] if preset else ["Python", "Java"]
d_fws = preset["frameworks"] if preset else ["React", "Pandas"]
d_certs = preset["verified_certs"] if preset else ["Python Certificate", "SQL Certificate"]

d_projects = preset["projects_count"] if preset else 3
d_complexity = preset["project_complexity"] if preset else 3
d_oss = preset["open_source"] if preset else 6
d_internships = preset["internships"] if preset else 1

d_apt = preset["aptitude_score"] if preset else 70
d_quant = preset["quant_score"] if preset else 72
d_logical = preset["logical_score"] if preset else 68
d_coding = preset["coding_score"] if preset else 65

d_comm = preset["comm_score"] if preset else 80
d_verbal = preset["verbal_score"] if preset else 78
d_interview = preset["interview_score"] if preset else 75
d_pres = preset["presentation_score"] if preset else 79

d_hacks = preset["hackathons"] if preset else 2
d_lead = preset["leadership"] if preset else 1
d_tech_soc = preset["tech_society"] if preset else 1
d_role = preset["target_role"] if preset else "Full Stack Developer"

# Student Profile Input Form
with st.form("student_profile_form"):
    st.subheader("📋 Enter Student Employability Profile")

    # Section A: Student Identity & Academic Profile
    st.markdown("#### A. Student Identity & Academic Profile")
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        student_name_input = st.text_input("Student Name *", value=d_name, help="Enter full name. Records with new names in department will be assigned IDs from STU1001 onwards.")

    a_col1, a_col2, a_col3, a_col4, a_col5, a_col6 = st.columns(6)
    with a_col1:
        cgpa = st.number_input("CGPA (5.0 - 10.0)", min_value=5.0, max_value=10.0, value=float(d_cgpa), step=0.01)
    with a_col2:
        tenth_pct = st.number_input("10th Percentage (%)", min_value=50.0, max_value=100.0, value=float(d_tenth), step=0.1)
    with a_col3:
        twelfth_pct = st.number_input("12th Percentage (%)", min_value=50.0, max_value=100.0, value=float(d_twelfth), step=0.1)
    with a_col4:
        backlogs = st.number_input("Active Backlogs (0-8)", min_value=0, max_value=8, value=int(d_backlogs), step=1)
    with a_col5:
        semester = st.selectbox("Current Semester", [5, 6, 7, 8], index=[5, 6, 7, 8].index(d_sem))
    with a_col6:
        department = st.selectbox("Department", ["CSE", "AI&DS", "IT", "ECE", "EEE", "Mechanical"], index=["CSE", "AI&DS", "IT", "ECE", "EEE", "Mechanical"].index(d_dept))

    # Section B: Programming & Technical Skills
    st.markdown("#### B. Programming & Technical Skills (0 = No Skill, 1 = Beginner, 2 = Intermediate, 3 = Advanced)")
    b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(6)
    with b_col1:
        python_skill = st.slider("Python Skill", 0, 3, int(d_py))
    with b_col2:
        java_skill = st.slider("Java Skill", 0, 3, int(d_java))
    with b_col3:
        sql_skill = st.slider("SQL Skill", 0, 3, int(d_sql))
    with b_col4:
        react_skill = st.slider("React Skill", 0, 3, int(d_react))
    with b_col5:
        aws_skill = st.slider("AWS Skill", 0, 3, int(d_aws))
    with b_col6:
        dsa_skill = st.slider("DSA Skill", 0, 3, int(d_dsa))

    b_sub1, b_sub2, b_sub3 = st.columns(3)
    with b_sub1:
        prog_langs = st.multiselect("Programming Languages Known", ["Python", "Java", "C++"], default=d_langs)
    with b_sub2:
        frameworks = st.multiselect("Frameworks & Tools Known", ["React", "Spring Boot", "Pandas", "AWS", "Git"], default=d_fws)
    with b_sub3:
        verified_certs = st.multiselect("Verified Certifications", ["Python Certificate", "SQL Certificate", "Cloud Certificate", "DSA Certificate", "Web Development Certificate"], default=d_certs)

    certifications_count = len(verified_certs)

    # Section C: Projects & Experience
    st.markdown("#### C. Projects & Experience")
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    with c_col1:
        projects_count = st.number_input("Projects Count (0-8)", min_value=0, max_value=8, value=int(d_projects), step=1)
    with c_col2:
        project_complexity = st.slider("Project Complexity (1-5)", 1, 5, int(d_complexity))
    with c_col3:
        open_source = st.number_input("Open Source Commits", min_value=0, max_value=30, value=int(d_oss), step=1)
    with c_col4:
        internships = st.number_input("Internships Completed (0-3)", min_value=0, max_value=3, value=int(d_internships), step=1)

    # Section D: Aptitude & Coding
    st.markdown("#### D. Aptitude & Coding Scores (40 - 100)")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        aptitude_score = st.number_input("Overall Aptitude Score", min_value=40, max_value=100, value=int(d_apt), step=1)
    with d_col2:
        quant_score = st.number_input("Quantitative Aptitude Score", min_value=40, max_value=100, value=int(d_quant), step=1)
    with d_col3:
        logical_score = st.number_input("Logical Aptitude Score", min_value=40, max_value=100, value=int(d_logical), step=1)
    with d_col4:
        coding_score = st.number_input("Coding Score", min_value=40, max_value=100, value=int(d_coding), step=1)

    # Section E: Communication & Soft Skills
    st.markdown("#### E. Communication & Interview Scores (40 - 100)")
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        comm_score = st.number_input("Communication Score", min_value=40, max_value=100, value=int(d_comm), step=1)
    with e_col2:
        verbal_score = st.number_input("Verbal Fluency Score", min_value=40, max_value=100, value=int(d_verbal), step=1)
    with e_col3:
        interview_score = st.number_input("Interview Score", min_value=40, max_value=100, value=int(d_interview), step=1)
    with e_col4:
        presentation_score = st.number_input("Presentation Score", min_value=40, max_value=100, value=int(d_pres), step=1)

    # Section F: Extracurricular & Target Role
    st.markdown("#### F. Extracurricular & Target Role")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        hackathons = st.number_input("Hackathons Participated (0-10)", min_value=0, max_value=10, value=int(d_hacks), step=1)
    with f_col2:
        leadership = st.number_input("Leadership Roles (0-5)", min_value=0, max_value=5, value=int(d_lead), step=1)
    with f_col3:
        tech_society = st.number_input("Tech Society Participation (0-5)", min_value=0, max_value=5, value=int(d_tech_soc), step=1)
    with f_col4:
        target_role = st.selectbox("Target Role", ["Full Stack Developer", "Data Analyst", "DevOps Engineer", "QA Engineer"], index=["Full Stack Developer", "Data Analyst", "DevOps Engineer", "QA Engineer"].index(d_role))

    submit_button = st.form_submit_button("🚀 PREDICT MY PLACEMENT")

if submit_button:
    # Friendly Input Bounds Validation
    errors = []
    clean_name = student_name_input.strip()
    if not clean_name:
        errors.append("Student Name is required.")
    if cgpa < 5.0 or cgpa > 10.0:
        errors.append("CGPA must be between 5.0 and 10.0")
    if tenth_pct < 50.0 or tenth_pct > 100.0:
        errors.append("10th percentage must be between 50% and 100%")
    if twelfth_pct < 50.0 or twelfth_pct > 100.0:
        errors.append("12th percentage must be between 50% and 100%")
    if backlogs < 0 or backlogs > 8:
        errors.append("Backlogs count must be between 0 and 8")

    if errors:
        for err in errors:
            st.error(f"⚠️ Validation Error: {err}")
    else:
        prog_str = "|".join(prog_langs) if prog_langs else "None"
        fw_str = "|".join(frameworks) if frameworks else "None"
        cert_str = "|".join(verified_certs) if verified_certs else "None"

        student_data = {
            "student_id": "STU_TEMP",
            "student_name": clean_name,
            "department": department,
            "semester": semester,
            "cgpa": cgpa,
            "tenth_percentage": tenth_pct,
            "twelfth_percentage": twelfth_pct,
            "backlogs": backlogs,
            "programming_languages": prog_str,
            "frameworks": fw_str,
            "verified_certifications": cert_str,
            "python_skill": python_skill,
            "java_skill": java_skill,
            "sql_skill": sql_skill,
            "react_skill": react_skill,
            "aws_skill": aws_skill,
            "dsa_skill": dsa_skill,
            "certifications_count": certifications_count,
            "projects_count": projects_count,
            "project_complexity": project_complexity,
            "open_source_contributions": open_source,
            "internships_count": internships,
            "aptitude_score": aptitude_score,
            "quantitative_aptitude_score": quant_score,
            "logical_aptitude_score": logical_score,
            "coding_score": coding_score,
            "communication_score": comm_score,
            "verbal_fluency_score": verbal_score,
            "interview_score": interview_score,
            "presentation_score": presentation_score,
            "hackathons_participated": hackathons,
            "leadership_roles": leadership,
            "tech_society_participation": tech_society,
            "target_role": target_role
        }

        try:
            with st.spinner("Processing profile through XGBoost pipeline..."):
                pred_res = predict_placement(student_data)

            prob_pct = pred_res['placement_probability']
            pred_label = pred_res['placement_prediction']
            readiness = pred_res['readiness_status']
            confidence = pred_res['confidence']
            X_instance = pred_res['X_trans_instance']

            # Compute Skill Gap & Roadmap
            gaps = analyze_skill_gap(student_data, target_role)
            roadmap = generate_personalized_roadmap(student_data, gaps, target_role)

            # PERSISTENCE TO DATASET & TPO DASHBOARD
            saved, save_msg, assigned_id = save_student_to_dataset(student_data, pred_res, gaps, roadmap)

            if saved:
                st.success(f"💾 **Dataset Persistence**: {save_msg}")
            else:
                st.info(f"ℹ️ **Dataset Persistence**: {save_msg}")

            st.markdown("## 🎯 Placement Prediction & Readiness Results")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Placement Probability</div>
                    <div class="metric-value" style="color: #3b82f6;">{prob_pct}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                pred_color = "#10b981" if pred_label == "Placed" else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Outcome</div>
                    <div class="metric-value" style="color: {pred_color};">{pred_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                badge_cls = "badge-ready" if readiness == "Ready" else ("badge-near-ready" if readiness == "Near Ready" else "badge-needs-training")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Readiness Status</div>
                    <div style="margin-top: 10px;"><span class="{badge_cls}">{readiness}</span></div>
                </div>
                """, unsafe_allow_html=True)

            with res_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value" style="color: #a855f7;">{confidence}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 2. SHAP EXPLAINER
            st.markdown("## 🔍 Why This Prediction? (SHAP Explainability)")
            st.caption("Feature-level contributions computed directly from the trained XGBoost model using SHAP TreeExplainer.")

            shap_exp = get_shap_explanation(X_instance)
            fig_shap = render_shap_bar_chart(shap_exp)

            shap_col1, shap_col2 = st.columns([1, 1])

            with shap_col1:
                st.pyplot(fig_shap)

            with shap_col2:
                st.markdown("#### 🌟 Key Driver Factors")
                
                st.markdown("##### 🟢 Top Positive Impact Factors (Increased Probability)")
                for pos in shap_exp['top_positives'][:4]:
                    st.success(f"**{pos['display_name']}**: +{pos['shap_value']:.3f} probability contribution")

                st.markdown("##### 🔴 Top Negative Impact Factors (Decreased Probability)")
                for neg in shap_exp['top_negatives'][:4]:
                    st.error(f"**{neg['display_name']}**: {neg['shap_value']:.3f} probability contribution")

            st.markdown("---")

            # 3. SKILL GAP ANALYZER
            st.markdown(f"## 📊 Target Role Skill Gap Analysis ({target_role})")
            st.caption(f"Benchmarking candidate's profile against standard industry entry requirements for {target_role}.")

            gap_df = pd.DataFrame(gaps)

            def color_priority(val):
                if val == 'Critical':
                    return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;'
                elif val == 'High':
                    return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: bold;'
                elif val == 'Medium':
                    return 'background-color: rgba(59, 130, 246, 0.2); color: #3b82f6;'
                else:
                    return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981;'

            styler = gap_df[['skill', 'category', 'current_level', 'required_level', 'gap', 'priority', 'status']].style
            if hasattr(styler, 'map'):
                styled_gap_df = styler.map(color_priority, subset=['priority'])
            else:
                styled_gap_df = styler.applymap(color_priority, subset=['priority'])

            st.dataframe(styled_gap_df, use_container_width=True)

            st.markdown("---")

            # 4. PERSONALIZED ROADMAP GENERATOR
            st.markdown(f"## 🗺️ Personalized Career Roadmap ({target_role})")
            st.caption("Tailored 4-Phase Action Plan structured based on identified skill gaps and target role requirements.")

            st.info(f"⏱️ **Estimated Completion Time**: **{roadmap['total_estimated_weeks']} Weeks** to reach Full Placement Readiness.")

            for phase in roadmap['phases']:
                with st.expander(f"📌 {phase['phase_name']} ({phase['duration']})", expanded=(phase['phase_num'] == 1)):
                    p_col1, p_col2 = st.columns([1, 1])
                    with p_col1:
                        st.markdown(f"**Focus Area**: {phase['focus']}")
                        st.markdown(f"**Key Skills to Improve**: {', '.join(phase['skills_to_improve'])}")
                        st.markdown("**Recommended Courses**:")
                        for c in phase['recommended_courses']:
                            st.markdown(f"- 🎓 {c}")
                    with p_col2:
                        st.markdown("**Actionable Tasks & Practice Drills**:")
                        for t in phase['action_tasks']:
                            st.markdown(f"- ✅ {t}")

        except Exception as e:
            st.error(f"⚠️ Prediction Engine Exception: {str(e)}")
            st.info("Ensure all model artifacts (models/xgboost_placement_model.pkl, models/preprocessor.pkl) are properly initialized.")
