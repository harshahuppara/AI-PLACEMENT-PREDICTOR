import streamlit as st

st.set_page_config(
    page_title="AI Placement Predictor | SIH",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS for Dark Glassmorphism Aesthetic
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
    }
    
    /* Glassmorphism Metric Card */
    .metric-card {
        background: rgba(30, 34, 45, 0.7);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 5px;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
        font-weight: 600;
    }
    
    /* Readiness Badges */
    .badge-ready {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-near-ready {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-needs-training {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* Feature Highlight Box */
    .feature-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    /* Primary Accent Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header
st.sidebar.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=60)
st.sidebar.title("AI Placement Predictor")
st.sidebar.caption("SIH 2026 AI Placement Predictor Engine")
st.sidebar.markdown("---")

# Navigation Guidance
st.sidebar.markdown("""
### 🧭 Navigation
- 🏠 **Home**: Platform Overview & System Flow
- 🎓 **Student Predictor**: Individual Assessment, SHAP & Roadmap
- 📊 **TPO Dashboard**: Institutional Placement Analytics
- 🔍 **Student Analysis**: Individual Inspection Search
- 🤖 **Model Explainability**: Model Transparency & Metrics
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Model**: XGBoost Classification Engine (52 Input Features, 5-Fold Stratified CV, Leakage-Free)")

# Header Banner
st.title("🎓 AI Placement Predictor & Employability Intelligence System")
st.subheader("Smart India Hackathon — End-to-End Multidimensional Employability Assessment")

st.markdown("""
Welcome to the **AI Placement Predictor Platform**. Built on a trained, leakage-free **XGBoost Classifier**, 
this system evaluates a student's **complete employability profile** across **52 parameters** to deliver precise placement predictions, SHAP explanations, role-specific skill gap analysis, and personalized career roadmaps.
""")

st.markdown("---")

# System Architecture & Flow
st.markdown("### ⚙️ System Architecture & Workflow")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🎓 Student Employability Flow</h4>
        <p><b>1. Student Profile Input:</b> Complete 52-feature profile collection across academics, technical skills, aptitude, communication, experience, and hackathons.</p>
        <p><b>2. XGBoost Inference Engine:</b> Predicts placement outcome, probability score (0-100%), and readiness status (Ready / Near Ready / Needs Training).</p>
        <p><b>3. SHAP Tree Explainer:</b> Identifies exact positive and negative factor contributions driving the model's prediction.</p>
        <p><b>4. Skill Gap Analyzer:</b> Benchmarks candidate profile against target role requirements.</p>
        <p><b>5. Personalized Roadmap:</b> Generates a 4-phase structured action plan with timeline & recommended courses.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>📊 TPO Institutional Analytics Flow</h4>
        <p><b>1. Batch Analytics:</b> Evaluates entire student cohort (1,000 students) across all departments.</p>
        <p><b>2. Readiness Distribution:</b> Real-time tracking of Ready, Near Ready, and Needs Training student counts.</p>
        <p><b>3. Vulnerable Students Identification:</b> Instant filtering of students requiring academic/training intervention.</p>
        <p><b>4. Institutional Skill Heatmap:</b> Department-wise skill deficit identification across programming, aptitude, and communication.</p>
        <p><b>5. Multi-Criterion Filters:</b> Dynamic filtering by Department, Semester, Target Role, and Placement Outcome.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Six Mandatory Feature Areas Section
st.markdown("### 🛡️ The 6 Mandatory Employability Feature Pillars")
st.caption("Our XGBoost model is trained on the complete student profile without relying solely on CGPA.")

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">💻</div>
        <div class="metric-label">Programming</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Python, Java, SQL, React, AWS, DSA</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">🎙️</div>
        <div class="metric-label">Interview</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Technical & Behavioral Interview Score</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">🧠</div>
        <div class="metric-label">Aptitude</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Quantitative & Logical Reasoning</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">🗣️</div>
        <div class="metric-label">Communication</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Verbal Fluency & Presentation</div>
    </div>
    """, unsafe_allow_html=True)

with m_col5:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">🛠️</div>
        <div class="metric-label">Projects</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Project Count & Complexity</div>
    </div>
    """, unsafe_allow_html=True)

with m_col6:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.8rem;">🏆</div>
        <div class="metric-label">Hackathons</div>
        <div style="font-size: 0.85rem; color: #9ca3af;">Hackathons Participated Count</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Navigation Buttons
st.markdown("### 🚀 Quick Access Modules")
q_col1, q_col2, q_col3, q_col4 = st.columns(4)

with q_col1:
    st.info("🎓 **Student Predictor**\n\nRun prediction, SHAP explanation & personalized roadmap.")

with q_col2:
    st.success("📊 **TPO Dashboard**\n\nView institutional placement metrics & skill heatmaps.")

with q_col3:
    st.warning("🔍 **Student Search**\n\nSearch and inspect individual student profiles.")

with q_col4:
    st.error("🤖 **Model Transparency**\n\nInspect XGBoost 5-fold CV metrics & feature importances.")
