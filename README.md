# 🧠 SkillMap AI — Intern Skill Gap Analysis

> *"Know your gaps. Bridge them. Get hired."*

---

## 📌 Project Overview

SkillMap AI analyzes the difference between **intern skills** and **industry-required skills** using real-world datasets and machine learning.

### 🔍 What it does:

* Extracts and cleans skills from resumes
* Compares them with industry job requirements
* Identifies missing (gap) skills
* Calculates a **gap score**
* Groups users into clusters using ML

---

## 📊 Datasets Used

| Dataset                            | Description                                  |
| ---------------------------------- | -------------------------------------------- |
| Data Science Job Postings & Skills | Real-world job postings with required skills |
| Resume Dataset                     | Intern and candidate resumes                 |

---

## ⚙️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn

  * TF-IDF Vectorization
  * K-Means Clustering
* KaggleHub (dataset access)

---

## 🔄 Workflow

### 1. Data Collection

* Download datasets using KaggleHub
* Load job postings, skills, and resumes

### 2. Data Cleaning

* Convert skills into structured lists
* Normalize text (lowercase, remove spaces)
* Remove empty records

### 3. Feature Engineering

* Convert skills into text format
* Apply **TF-IDF Vectorization**
* Build a unified vocabulary

### 4. Gap Analysis

* Identify **top 20 industry skills**
* Compare each intern’s skills with industry demand
* Compute:

  * Missing skills (Gap Skills)
  * Gap Score (%)

### 5. Clustering

* Use **K-Means (K=5)** to group interns
* Each cluster represents a skill gap category

### 6. Output

* Final dataset saved as:

```
intern_gap_analysis.csv
```

---

## 📈 Key Output Fields

| Column     | Description          |
| ---------- | -------------------- |
| Job_Role   | Candidate role       |
| Gap_Skills | Missing skills       |
| Gap_Score  | Percentage gap       |
| Cluster    | Assigned skill group |

---

## 🚀 How To Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn kagglehub
```

### 2. Run Training Pipeline

```bash
python train.py
```

### 3. Output

* Generated file:

```
intern_gap_analysis.csv
```

---

## 💡 Insights

* Most interns lack critical industry skills
* High gap scores indicate strong mismatch with job market
* Clustering helps identify common learning paths

---

## 📌 Future Improvements

* Add recommendation system (courses for each gap)
* Build interactive UI (Gradio)
* Deploy on Hugging Face Spaces
* Improve skill extraction using NLP

---

## 👤 Author

**MAhmadMakhdoom**
Software Engineering Student | AI Enthusiast

---
