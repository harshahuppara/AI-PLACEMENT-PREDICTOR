ROLE_BENCHMARKS = {
    "Full Stack Developer": {
        "Python": {"req": 2, "category": "Programming"},
        "Java": {"req": 2, "category": "Programming"},
        "SQL": {"req": 2, "category": "Programming"},
        "React": {"req": 3, "category": "Programming"},
        "AWS": {"req": 1, "category": "Programming"},
        "Data Structures & Algorithms": {"req": 2, "category": "Programming"},
        "Coding Performance": {"req": 70, "category": "Aptitude & Coding"},
        "General Aptitude": {"req": 70, "category": "Aptitude & Coding"},
        "Interview Performance": {"req": 75, "category": "Interview"},
        "Communication & Soft Skills": {"req": 75, "category": "Communication"},
        "Projects Count": {"req": 3, "category": "Projects"},
        "Project Complexity": {"req": 3, "category": "Projects"},
        "Hackathons Participated": {"req": 1, "category": "Hackathons"}
    },
    "Data Analyst": {
        "Python": {"req": 3, "category": "Programming"},
        "Java": {"req": 1, "category": "Programming"},
        "SQL": {"req": 3, "category": "Programming"},
        "React": {"req": 0, "category": "Programming"},
        "AWS": {"req": 1, "category": "Programming"},
        "Data Structures & Algorithms": {"req": 1, "category": "Programming"},
        "Coding Performance": {"req": 65, "category": "Aptitude & Coding"},
        "General Aptitude": {"req": 80, "category": "Aptitude & Coding"},
        "Interview Performance": {"req": 70, "category": "Interview"},
        "Communication & Soft Skills": {"req": 75, "category": "Communication"},
        "Projects Count": {"req": 2, "category": "Projects"},
        "Project Complexity": {"req": 3, "category": "Projects"},
        "Hackathons Participated": {"req": 0, "category": "Hackathons"}
    },
    "DevOps Engineer": {
        "Python": {"req": 2, "category": "Programming"},
        "Java": {"req": 1, "category": "Programming"},
        "SQL": {"req": 2, "category": "Programming"},
        "React": {"req": 1, "category": "Programming"},
        "AWS": {"req": 3, "category": "Programming"},
        "Data Structures & Algorithms": {"req": 2, "category": "Programming"},
        "Coding Performance": {"req": 70, "category": "Aptitude & Coding"},
        "General Aptitude": {"req": 75, "category": "Aptitude & Coding"},
        "Interview Performance": {"req": 75, "category": "Interview"},
        "Communication & Soft Skills": {"req": 70, "category": "Communication"},
        "Projects Count": {"req": 3, "category": "Projects"},
        "Project Complexity": {"req": 3, "category": "Projects"},
        "Hackathons Participated": {"req": 1, "category": "Hackathons"}
    },
    "QA Engineer": {
        "Python": {"req": 2, "category": "Programming"},
        "Java": {"req": 2, "category": "Programming"},
        "SQL": {"req": 2, "category": "Programming"},
        "React": {"req": 1, "category": "Programming"},
        "AWS": {"req": 1, "category": "Programming"},
        "Data Structures & Algorithms": {"req": 1, "category": "Programming"},
        "Coding Performance": {"req": 60, "category": "Aptitude & Coding"},
        "General Aptitude": {"req": 65, "category": "Aptitude & Coding"},
        "Interview Performance": {"req": 75, "category": "Interview"},
        "Communication & Soft Skills": {"req": 80, "category": "Communication"},
        "Projects Count": {"req": 2, "category": "Projects"},
        "Project Complexity": {"req": 2, "category": "Projects"},
        "Hackathons Participated": {"req": 0, "category": "Hackathons"}
    }
}

def analyze_skill_gap(student_profile, target_role="Full Stack Developer"):
    """
    Compares student's actual attributes against target role requirements.
    Returns structured list of gaps with priority and category.
    """
    benchmarks = ROLE_BENCHMARKS.get(target_role, ROLE_BENCHMARKS["Full Stack Developer"])

    # Extract student values
    student_curr = {
        "Python": student_profile.get("python_skill", 0),
        "Java": student_profile.get("java_skill", 0),
        "SQL": student_profile.get("sql_skill", 0),
        "React": student_profile.get("react_skill", 0),
        "AWS": student_profile.get("aws_skill", 0),
        "Data Structures & Algorithms": student_profile.get("dsa_skill", 0),
        "Coding Performance": student_profile.get("coding_score", 0),
        "General Aptitude": student_profile.get("aptitude_score", 0),
        "Interview Performance": student_profile.get("interview_score", 0),
        "Communication & Soft Skills": student_profile.get("communication_score", 0),
        "Projects Count": student_profile.get("projects_count", 0),
        "Project Complexity": student_profile.get("project_complexity", 1),
        "Hackathons Participated": student_profile.get("hackathons_participated", 0)
    }

    gaps = []
    for skill_name, config in benchmarks.items():
        req_val = config["req"]
        curr_val = student_curr.get(skill_name, 0)
        category = config["category"]

        if req_val == 0:
            gap_amt = 0
            priority = "On Track"
            status = "Achieved"
        elif "Score" in skill_name or "Performance" in skill_name:
            gap_amt = max(0, req_val - curr_val)
            if gap_amt >= 20:
                priority = "Critical"
                status = "Needs Urgent Work"
            elif gap_amt >= 10:
                priority = "High"
                status = "Moderate Gap"
            elif gap_amt > 0:
                priority = "Medium"
                status = "Slight Gap"
            else:
                priority = "On Track"
                status = "Achieved"
        else:
            gap_amt = max(0, req_val - curr_val)
            if gap_amt >= 2:
                priority = "Critical"
                status = "Needs Urgent Work"
            elif gap_amt == 1:
                priority = "High"
                status = "Moderate Gap"
            else:
                priority = "On Track"
                status = "Achieved"

        gaps.append({
            "skill": skill_name,
            "category": category,
            "current_level": curr_val,
            "required_level": req_val,
            "gap": gap_amt,
            "priority": priority,
            "status": status
        })

    gaps_sorted = sorted(gaps, key=lambda x: (
        0 if x['priority'] == 'Critical' else (1 if x['priority'] == 'High' else (2 if x['priority'] == 'Medium' else 3))
    ))

    return gaps_sorted
