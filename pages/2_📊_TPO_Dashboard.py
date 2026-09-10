import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.model_loader import load_dataset
from utils.dashboard import get_institutional_data, get_vulnerable_students, get_skill_heatmap_data

st.set_page_config(page_title="TPO Institutional Dashboard | SIH", page_icon="📊", layout="wide")

st.title("📊 Training & Placement Officer (TPO) Institutional Dashboard")
st.caption("Batch Placement Analytics, Department Readiness Breakdown, Vulnerable Students Identification & Skill Heatmaps")

st.markdown("---")

df_raw = load_dataset()

# Sidebar Filters
st.sidebar.header("🔍 Dashboard Filters")

# Reset Filter Mechanism
if "tpo_dept" not in st.session_state:
    st.session_state.tpo_dept = "All"
if "tpo_sem" not in st.session_state:
    st.session_state.tpo_sem = "All"
if "tpo_role" not in st.session_state:
    st.session_state.tpo_role = "All"
if "tpo_readiness" not in st.session_state:
    st.session_state.tpo_readiness = "All"
if "tpo_outcome" not in st.session_state:
    st.session_state.tpo_outcome = "All"

def reset_filters():
    st.session_state.tpo_dept = "All"
    st.session_state.tpo_sem = "All"
    st.session_state.tpo_role = "All"
    st.session_state.tpo_readiness = "All"
    st.session_state.tpo_outcome = "All"

st.sidebar.button("🔄 Reset Filters", on_click=reset_filters)

depts = ["All"] + sorted(df_raw['department'].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", depts, key="tpo_dept")

semesters = ["All"] + sorted(df_raw['semester'].unique().tolist())
selected_sem = st.sidebar.selectbox("Semester", semesters, key="tpo_sem")

roles = ["All"] + sorted(df_raw['target_role'].unique().tolist())
selected_role = st.sidebar.selectbox("Target Role", roles, key="tpo_role")

readiness_opts = ["All", "Ready", "Near Ready", "Needs Training"]
selected_readiness = st.sidebar.selectbox("Readiness Status", readiness_opts, key="tpo_readiness")

outcomes = ["All", "Placed", "Not Placed"]
selected_outcome = st.sidebar.selectbox("Placement Outcome", outcomes, key="tpo_outcome")

# Filter Data
filtered_df = df_raw.copy()
if selected_dept != "All":
    filtered_df = filtered_df[filtered_df['department'] == selected_dept]
if selected_sem != "All":
    filtered_df = filtered_df[filtered_df['semester'] == int(selected_sem)]
if selected_role != "All":
    filtered_df = filtered_df[filtered_df['target_role'] == selected_role]
if selected_readiness != "All":
    filtered_df = filtered_df[filtered_df['placement_status'] == selected_readiness]
if selected_outcome != "All":
    filtered_df = filtered_df[filtered_df['placement_outcome'] == selected_outcome]

res = get_institutional_data(filtered_df)
summary, dept_stats = res[0], res[1]

if len(filtered_df) == 0:
    st.warning("No student records match the selected filter criteria. Click 'Reset Filters' to restore default dataset view.")
else:
    # 1. TPO SUMMARY METRICS
    st.markdown("### 📈 Institutional Placement Summary")
    
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric("Total Students", summary['total_students'])
    with m2:
        st.metric("Placed Count", summary['placed_count'], f"{summary['placed_pct']}% Placed")
    with m3:
        st.metric("Not Placed", summary['not_placed_count'])
    with m4:
        st.metric("Ready Count (>=80%)", summary['ready_count'])
    with m5:
        st.metric("Near Ready (60-79%)", summary['near_ready_count'])
    with m6:
        st.metric("Needs Training (<60%)", summary['needs_training_count'])
    with m7:
        st.metric("Avg Placement Prob", f"{summary['avg_probability']}%")

    st.markdown("#### 🎯 Cohort Employability Averages")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1:
        st.metric("Avg CGPA", summary['avg_cgpa'])
    with a2:
        st.metric("Avg Aptitude", summary['avg_aptitude'])
    with a3:
        st.metric("Avg Interview", summary['avg_interview'])
    with a4:
        st.metric("Avg Communication", summary['avg_communication'])
    with a5:
        st.metric("Avg Projects", summary['avg_projects'])
    with a6:
        st.metric("Avg Hackathons", summary['avg_hackathons'])

    st.markdown("---")

    # 2. DEPARTMENT ANALYTICS
    st.markdown("### 🏢 Department-Wise Placement & Readiness Analytics")

    dept_col1, dept_col2 = st.columns([1, 1])

    with dept_col1:
        st.markdown("#### Average Placement Probability by Department")
        if len(dept_stats) > 0:
            fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0e1117')
            ax.set_facecolor('#1e222d')
            bars = ax.bar(dept_stats['department'], dept_stats['avg_probability'], color='#3b82f6', width=0.5)
            ax.set_ylabel("Avg Probability (%)", color='#9ca3af')
            ax.set_ylim(0, 100)
            ax.tick_params(colors='#ffffff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#374151')
            ax.spines['left'].set_color('#374151')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5, f"{height:.1f}%", ha='center', va='bottom', color='#ffffff', fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    with dept_col2:
        st.markdown("#### Readiness Status Distribution by Department")
        if len(dept_stats) > 0:
            dept_chart_df = dept_stats.set_index('department')[['ready_count', 'near_ready_count', 'needs_training_count']]
            dept_chart_df.columns = ['Ready', 'Near Ready', 'Needs Training']
            st.bar_chart(dept_chart_df, color=['#10b981', '#f59e0b', '#ef4444'])

    st.markdown("---")

    # 3. VULNERABLE STUDENTS SECTION
    st.markdown("### 🚨 Students Needing Attention (Low Placement Readiness)")
    st.caption("Students identified with placement probability < 60% or marked as 'Needs Training'. Allows direct TPO intervention.")

    vulnerable_df = get_vulnerable_students(filtered_df)
    st.markdown(f"**Identified Needing Attention**: **{len(vulnerable_df)} Students**")

    if len(vulnerable_df) > 0:
        st.dataframe(vulnerable_df, use_container_width=True)
    else:
        st.success("No vulnerable students found under current filter criteria!")

    st.markdown("---")

    # 4. INSTITUTIONAL SKILL DEFICIT HEATMAP
    st.markdown("### 🗺️ Institutional Skill Deficit Heatmap")
    st.caption("Department-level average skill mastery ratings (normalized to 0-100 scale). Identifies institutional training priorities.")

    heatmap_df = get_skill_heatmap_data(filtered_df)

    fig_hm, ax_hm = plt.subplots(figsize=(10, 4.5), facecolor='#0e1117')
    ax_hm.set_facecolor('#1e222d')

    cax = ax_hm.matshow(heatmap_df.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(cax)

    ax_hm.set_xticks(range(len(heatmap_df.columns)))
    ax_hm.set_yticks(range(len(heatmap_df.index)))
    ax_hm.set_xticklabels(heatmap_df.columns, rotation=45, ha='left', color='#ffffff')
    ax_hm.set_yticklabels(heatmap_df.index, color='#ffffff')

    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.iloc[i, j]
            ax_hm.text(j, i, f"{val:.0f}", ha='center', va='center', color='black' if val > 60 else 'white', fontweight='bold', fontsize=9)

    plt.title("Department Skill Deficit Matrix (0 = Deficit, 100 = Mastery)", color='#ffffff', fontsize=12, pad=30, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_hm)
