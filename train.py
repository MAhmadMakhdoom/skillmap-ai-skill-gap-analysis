
# ============================================================
# SkillMap AI — Training Pipeline
# ============================================================

import kagglehub
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

print("🚀 Starting Training...")

# STEP 1: Download datasets
path1 = kagglehub.dataset_download("asaniczka/data-science-job-postings-and-skills")
path2 = kagglehub.dataset_download("saugataroyarghya/resume-dataset")

# STEP 2: Load data
df_jobs    = pd.read_csv(f"{path1}/job_postings.csv")
df_skills  = pd.read_csv(f"{path1}/job_skills.csv")
df_resumes = pd.read_csv(f"{path2}/resume_data.csv")

df_intern = df_resumes[["\ufeffjob_position_name", "skills"]].copy()
df_intern.columns = ["Job_Role", "Intern_Skills"]

df_industry = df_jobs[["job_link","job_title"]].merge(df_skills, on="job_link", how="left")
df_industry = df_industry[["job_title","job_skills"]].copy()
df_industry.columns = ["Job_Role","Industry_Skills"]

print("✅ Data Loaded")

# STEP 3: Cleaning
def clean_intern(s):
    try:
        return [x.strip().lower() for x in ast.literal_eval(s)]
    except:
        return []

def clean_industry(s):
    try:
        return [x.strip().lower() for x in s.split(",")]
    except:
        return []

df_intern["Intern_Skills"] = df_intern["Intern_Skills"].apply(clean_intern)
df_industry["Industry_Skills"] = df_industry["Industry_Skills"].apply(clean_industry)

df_intern   = df_intern[df_intern["Intern_Skills"].map(len) > 0]
df_industry = df_industry[df_industry["Industry_Skills"].map(len) > 0]

# STEP 4: TF-IDF
df_intern["text"]   = df_intern["Intern_Skills"].apply(lambda x: " ".join(x))
df_industry["text"] = df_industry["Industry_Skills"].apply(lambda x: " ".join(x))

all_text = pd.concat([df_intern["text"], df_industry["text"]])

tfidf = TfidfVectorizer()
tfidf.fit(all_text)

print("✅ TF-IDF Done")

# STEP 5: Gap Analysis
all_skills = []
for s in df_industry["Industry_Skills"]:
    all_skills.extend(s)

top_20 = [s for s, c in Counter(all_skills).most_common(20)]

gaps, scores = [], []

for skills in df_intern["Intern_Skills"]:
    gap = list(set(top_20) - set(skills))
    score = round(len(gap)/len(top_20)*100,2)
    gaps.append(gap)
    scores.append(score)

df_intern["Gap_Skills"] = gaps
df_intern["Gap_Score"]  = scores

print("✅ Gap Analysis Done")

# STEP 6: Clustering
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_intern["Gap_Skills"])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_intern["Cluster"] = kmeans.fit_predict(X)

print("✅ Clustering Done")

# STEP 7: Save
df_intern.to_csv("intern_gap_analysis.csv", index=False)

print("🎯 Training Complete!")
