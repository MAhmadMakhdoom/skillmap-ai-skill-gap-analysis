
import gradio as gr
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROLE_SKILLS = {
    "Data Scientist": {
        "skills": ["python", "machine learning", "deep learning",
                   "statistics", "sql", "data visualization", "tensorflow", "pytorch"],
        "weights": {"python": 3, "machine learning": 3, "deep learning": 3,
                    "statistics": 2, "sql": 2, "data visualization": 2,
                    "tensorflow": 2, "pytorch": 2}
    },
    "Data Analyst": {
        "skills": ["excel", "sql", "data visualization", "python",
                   "tableau", "power bi", "statistics"],
        "weights": {"excel": 3, "sql": 3, "data visualization": 3,
                    "python": 2, "tableau": 2, "power bi": 2, "statistics": 2}
    },
    "ML Engineer": {
        "skills": ["python", "machine learning", "deep learning", "mlops",
                   "docker", "kubernetes", "aws", "tensorflow", "pytorch"],
        "weights": {"python": 3, "machine learning": 3, "deep learning": 3,
                    "mlops": 2, "docker": 2, "kubernetes": 2, "aws": 2}
    },
    "Software Engineer": {
        "skills": ["python", "java", "sql", "git", "docker",
                   "aws", "system design", "data structures", "algorithms"],
        "weights": {"python": 3, "java": 3, "sql": 2, "git": 2,
                    "docker": 2, "aws": 2, "system design": 3}
    },
    "Data Engineer": {
        "skills": ["python", "sql", "spark", "hadoop", "aws",
                   "data warehousing", "etl", "kafka", "airflow"],
        "weights": {"python": 3, "sql": 3, "spark": 3, "hadoop": 2,
                    "aws": 2, "data warehousing": 2, "etl": 3}
    }
}

RECOMMENDATIONS = {
    "Data Scientist":   ["Machine Learning A-Z (Udemy)", "Deep Learning Specialization (Coursera)",
                         "Statistics for Data Science", "SQL for Data Science"],
    "Data Analyst":     ["Excel Advanced Analytics", "Tableau Desktop Specialist",
                         "SQL Bootcamp", "Data Visualization with Python"],
    "ML Engineer":      ["MLOps Fundamentals", "Docker & Kubernetes for ML",
                         "AWS Machine Learning Specialty", "PyTorch for Deep Learning"],
    "Software Engineer":["System Design Interview Prep", "Data Structures & Algorithms",
                         "Docker for Developers", "AWS Cloud Practitioner"],
    "Data Engineer":    ["Apache Spark for Big Data", "Data Warehousing Fundamentals",
                         "Airflow for Data Pipelines", "Kafka for Data Engineering"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text

def extract_skills(text, skills_db):
    found = set()
    for skill in skills_db:
        if skill in text:
            found.add(skill)
    return found

def weighted_gap(job_skills, missing_skills, weights):
    total_weight   = sum(weights.get(s, 1) for s in job_skills)
    missing_weight = sum(weights.get(s, 1) for s in missing_skills)
    if total_weight == 0:
        return 0
    return round((missing_weight / total_weight) * 100, 2)

def analyze_gap_gradio(resume_text, role):
    if not resume_text:
        return "Please provide your skills!", "", "", "", ""

    resume_clean  = clean_text(resume_text)
    skills_db     = ROLE_SKILLS[role]["skills"]
    weights       = ROLE_SKILLS[role]["weights"]
    resume_skills = extract_skills(resume_clean, skills_db)
    job_skills    = set(skills_db)
    missing       = job_skills - resume_skills

    vectorizer  = TfidfVectorizer()
    job_text    = " ".join(skills_db)
    vectors     = vectorizer.fit_transform([resume_clean, job_text])
    similarity  = cosine_similarity(vectors[0], vectors[1])[0][0]
    match_score = round(similarity * 100, 2)
    gap_score   = weighted_gap(job_skills, missing, weights)

    have_str    = ", ".join(sorted(resume_skills)) if resume_skills else "None"
    missing_str = ", ".join(sorted(missing)) if missing else "None - You are ready!"
    courses_str = "\n".join([f"- {c}" for c in RECOMMENDATIONS[role]])

    summary = f"""
## Results for {role}
| Metric | Score |
|--------|-------|
| Match Score | {match_score}% |
| Gap Score   | {gap_score}%  |
| Skills Have | {len(resume_skills)} |
| Missing     | {len(missing)} |
"""
    return summary, have_str, missing_str, courses_str, f"{match_score}% Match | {gap_score}% Gap"

with gr.Blocks(
    title="SkillMap AI",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue", neutral_hue="slate")
) as demo:

    gr.Markdown("# Intern Skill Gap Analyzer")
    gr.Markdown("Identify skill gaps and get personalized training recommendations")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            resume_input = gr.Textbox(
                label="Your Skills / Resume Text",
                placeholder="e.g. Python, SQL, pandas, numpy, data analysis...",
                lines=8
            )
        with gr.Column(scale=1):
            role_selector = gr.Dropdown(
                label="Target Role",
                choices=list(ROLE_SKILLS.keys()),
                value="Data Scientist"
            )
            analyze_btn = gr.Button("Analyze My Skill Gap", variant="primary", size="lg")

    gr.Markdown("---")
    score_out   = gr.Markdown()
    with gr.Row():
        have_out    = gr.Textbox(label="Skills You Have", lines=3)
        missing_out = gr.Textbox(label="Missing Skills",  lines=3)
    courses_out = gr.Textbox(label="Recommended Courses", lines=6)
    status_out  = gr.Textbox(label="Quick Summary")

    analyze_btn.click(
        fn=analyze_gap_gradio,
        inputs=[resume_input, role_selector],
        outputs=[score_out, have_out, missing_out, courses_out, status_out]
    )

demo.launch(share=True)
