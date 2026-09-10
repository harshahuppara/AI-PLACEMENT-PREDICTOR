import streamlit as st
import pandas as pd
from utils.model_loader import load_dataset
from utils.prediction import predict_placement
from utils.shap_explainer import get_shap_explanation, render_shap_bar_chart
from utils.skill_gap import analyze_skill_gap
from utils.roadmap import generate_personalized_roadmap

st.set_page_config(page_title="Individual Student Inspection | SIH", page_icon="🔍", layout="wide")

st.title("🔍 Individual Student Inspection & Analysis")
st.caption("Search student database by Student ID or Name to inspect individual profile, SHAP explanations, skill gaps, and roadmap.")

st.markdown("---")

df_dataset = load_dataset()

# Student Search Filter
s_col1, s_col2 = st.columns([1, 2])

with s_col1:
    search_query = st.text_input("🔎 Search Student ID or Name", value="STU0001")

matches = df_dataset[
    (df_dataset['student_id'].str.contains(search_query, case=False, na=False)) |
    (df_dataset['student_name'].str.contains(search_query, case=False, na=False))
]

if len(matches) == 0:
    st.error(f"No student matching '{search_query}' found in dataset.")
else:
    selected_student_id = st.selectbox(
        "Select Matching Student",
        options=matches['student_id'].tolist(),
        format_func=lambda x: f"{x} - {matches[matches['student_id']==x]['student_name'].values[0]} ({matches[matches['student_id']==x]['department'].values[0]})"
    )

    student_row = matches[matches['student_id'] == selected_student_id].iloc[0]
    student_dict = student_row.to_dict()

    st.markdown("---")

    # 1. PROFILE SUMMARY CARD
    st.markdown(f"### 👤 Profile Summary: {student_dict['student_name']} ({student_dict['student_id']})")
    
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        st.markdown(f"**Department**\n\n{student_dict['department']}")
    with p2:
        st.markdown(f"**Semester**\n\nSem {student_dict['semester']}")
    with p3:
        st.markdown(f"**CGPA**\n\n{student_dict['cgpa']}")
    with p4:
        st.markdown(f"**Backlogs**\n\n{student_dict['backlogs']}")
    with p5:
        st.markdown(f"**Target Role**\n\n{student_dict['target_role']}")
    with p6:
        st.markdown(f"**Projects**\n\n{student_dict['projects_count']} Projects")

    # 2. RUN MODEL PREDICTION FOR INSPECTED STUDENT
    pred_res = predict_placement(student_dict)

    prob_pct = pred_res['placement_probability']
    pred_label = pred_res['placement_prediction']
    readiness = pred_res['readiness_status']
    confidence = pred_res['confidence']
    X_instance = pred_res['X_trans_instance']

    st.markdown("#### 🎯 Model Prediction Overview")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Placement Probability</div>
            <div class="metric-value" style="color: #3b82f6;">{prob_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        pred_color = "#10b981" if pred_label == "Placed" else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Outcome</div>
            <div class="metric-value" style="color: {pred_color};">{pred_label}</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        badge_cls = "badge-ready" if readiness == "Ready" else ("badge-near-ready" if readiness == "Near Ready" else "badge-needs-training")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Readiness Status</div>
            <div style="margin-top: 10px;"><span class="{badge_cls}">{readiness}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value" style="color: #a855f7;">{confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for Inspection Sub-sections
    tab1, tab2, tab3 = st.tabs(["🤖 SHAP Explanation", "📊 Skill Gap Analysis", "🗺️ Personalized Roadmap"])

    with tab1:
        st.markdown("#### SHAP Feature Contributions for Inspected Student")
        shap_exp = get_shap_explanation(X_instance)
        fig_shap = render_shap_bar_chart(shap_exp)

        sh_col1, sh_col2 = st.columns([1, 1])
        with sh_col1:
            st.pyplot(fig_shap)
        with sh_col2:
            st.markdown("##### 🟢 Key Positive Factors")
            for pos in shap_exp['top_positives'][:4]:
                st.success(f"**{pos['display_name']}**: +{pos['shap_value']:.3f}")

            st.markdown("##### 🔴 Key Negative Factors")
            for neg in shap_exp['top_negatives'][:4]:
                st.error(f"**{neg['display_name']}**: {neg['shap_value']:.3f}")

    with tab2:
        st.markdown(f"#### Skill Gap Analysis vs {student_dict['target_role']} Requirements")
        gaps = analyze_skill_gap(student_dict, student_dict['target_role'])
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
            styled_gap = styler.map(color_priority, subset=['priority'])
        else:
            styled_gap = styler.applymap(color_priority, subset=['priority'])
        st.dataframe(styled_gap, use_container_width=True)

    with tab3:
        st.markdown("#### Personalized Action Plan")
        gaps = analyze_skill_gap(student_dict, student_dict['target_role'])
        roadmap = generate_personalized_roadmap(student_dict, gaps, student_dict['target_role'])

        st.info(f"⏱️ **Estimated Duration**: **{roadmap['total_estimated_weeks']} Weeks** to reach Full Readiness.")

        for phase in roadmap['phases']:
            with st.expander(f"📌 {phase['phase_name']} ({phase['duration']})"):
                st.markdown(f"**Focus**: {phase['focus']}")
                st.markdown(f"**Courses**: {', '.join(phase['recommended_courses'])}")
                st.markdown("**Tasks**:")
                for task in phase['action_tasks']:
                    st.markdown(f"- ✅ {task}")
