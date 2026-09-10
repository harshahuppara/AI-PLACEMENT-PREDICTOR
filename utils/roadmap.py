def generate_personalized_roadmap(student_profile, skill_gaps, target_role="Full Stack Developer"):
    """
    Generates a personalized 4-Phase Roadmap based on target role, skill gaps, and current student levels.
    """
    critical_gaps = [g['skill'] for g in skill_gaps if g['priority'] == 'Critical']
    high_gaps = [g['skill'] for g in skill_gaps if g['priority'] == 'High']
    all_gaps = [g['skill'] for g in skill_gaps if g['gap'] > 0]

    # Role specific recommendations
    role_course_map = {
        "Full Stack Developer": [
            "Meta Front-End Developer Professional Certificate",
            "Java Programming & Spring Boot Masterclass",
            "Data Structures & Algorithms in Java/Python",
            "MongoDB & Node.js Backend Architecture",
            "Full Stack Web Development Capstone"
        ],
        "Data Analyst": [
            "Python for Data Science & Data Analysis",
            "SQL & Database Fundamentals for Analytics",
            "Statistics for Data Science & Predictive Modeling",
            "Microsoft Power BI Data Analyst Certification",
            "Advanced Excel & Tableau Dashboard Mastery"
        ],
        "DevOps Engineer": [
            "AWS Cloud Practitioner & Solutions Architect",
            "Docker & Kubernetes Operations Bootcamp",
            "Linux System Administration & Shell Scripting",
            "CI/CD Pipelines with GitHub Actions & Jenkins",
            "Terraform Infrastructure as Code"
        ],
        "QA Engineer": [
            "Java & Selenium Automation Testing Frameworks",
            "Postman & REST API Automated Testing",
            "Manual Testing Foundations & Test Case Design",
            "Mastering Test Automation & DevOps Integration",
            "Cypress & Playwright Modern E2E Testing"
        ]
    }

    courses = role_course_map.get(target_role, role_course_map["Full Stack Developer"])

    # Estimate timeline based on number of critical/high gaps
    base_weeks = 8
    total_weeks = min(12, max(6, base_weeks + len(critical_gaps) * 2 + len(high_gaps)))

    w_p1 = max(2, round(total_weeks * 0.25))
    w_p2 = max(3, round(total_weeks * 0.30))
    w_p3 = max(2, round(total_weeks * 0.25))
    w_p4 = max(2, round(total_weeks * 0.20))

    roadmap = {
        "target_role": target_role,
        "total_estimated_weeks": total_weeks,
        "phases": [
            {
                "phase_num": 1,
                "phase_name": "PHASE 1 — FOUNDATION & CORE CONCEPTS",
                "duration": f"Weeks 1–{w_p1}",
                "focus": "Strengthening fundamental core subjects, math aptitude, and baseline coding skills.",
                "skills_to_improve": critical_gaps[:2] if critical_gaps else ["DSA & Aptitude Fundamentals"],
                "recommended_courses": courses[:2],
                "action_tasks": [
                    "Complete daily 45-minute quantitative & logical reasoning drills.",
                    "Review object-oriented programming fundamentals and data structures.",
                    "Solve 3 baseline coding problems daily on LeetCode/HackerRank."
                ]
            },
            {
                "phase_num": 2,
                "phase_name": "PHASE 2 — DEEP SKILL DEVELOPMENT & FRAMEWORKS",
                "duration": f"Weeks {w_p1+1}–{w_p1+w_p2}",
                "focus": f"Building specialized technical competence required for {target_role}.",
                "skills_to_improve": (critical_gaps[2:] + high_gaps)[:3] if (critical_gaps or high_gaps) else [f"Advanced {target_role} Tools"],
                "recommended_courses": courses[2:4],
                "action_tasks": [
                    f"Complete hands-on modules in {courses[2] if len(courses)>2 else 'Advanced Tech'}.",
                    "Implement 5 medium-complexity algorithmic problems weekly.",
                    "Participate in weekly online mock aptitude tests and timed coding contests."
                ]
            },
            {
                "phase_num": 3,
                "phase_name": "PHASE 3 — CAPSTONE PROJECT BUILDING & EXPERIENCE",
                "duration": f"Weeks {w_p1+w_p2+1}–{w_p1+w_p2+w_p3}",
                "focus": "Constructing high-impact industry-ready projects and open-source contributions.",
                "skills_to_improve": ["Production Deployment", "Full-Stack System Architecture", "Git & Portfolio"],
                "recommended_courses": [courses[-1]],
                "action_tasks": [
                    f"Build & deploy a full-featured capstone project relevant to {target_role}.",
                    "Publish code repositories to GitHub with documentation & architecture diagrams.",
                    "Contribute to open-source repositories and participate in a weekend hackathon."
                ]
            },
            {
                "phase_num": 4,
                "phase_name": "PHASE 4 — INTERVIEW MASTERY & PLACEMENT DRILLS",
                "duration": f"Weeks {w_p1+w_p2+w_p3+1}–{total_weeks}",
                "focus": "Mock interviews, behavioral fluency, presentation practice, and company placement tests.",
                "skills_to_improve": ["Interview Readiness", "Verbal Fluency", "Presentation Skills"],
                "recommended_courses": ["Placement Interview Mastery & Behavioral Drill Suite"],
                "action_tasks": [
                    "Conduct 5 peer-to-peer technical mock interviews with STAR framework responses.",
                    "Practice 1-minute elevator pitch and project walkthrough presentations.",
                    "Review target company past placement papers and system design questions."
                ]
            }
        ]
    }

    return roadmap
