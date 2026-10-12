# DecodeLabs AI Internship - Project 3: AI Recommendation Logic
# Capstone: Tech Stack Recommender (Content-Based Filtering)
#
# Pipeline (IPO):
#   INPUT   -> user enters 3+ skills (Ingestion)
#   PROCESS -> TF-IDF vectors + Cosine Similarity (Scoring)
#   OUTPUT  -> sorted, filtered Top-N career roles (Sorting + Filtering)

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "raw_skills.csv"   # optional: columns "role" and "skills" (skills separated by ";")
MIN_SKILLS = 3
TOP_N = 3

# ---------------------------------------------------------
# 1. Dataset Setup (items = job roles, attributes = skills)
# ---------------------------------------------------------
BUILT_IN_ROLES = {
    "Data Scientist": "python; sql; machine learning; data analysis; statistics; pandas; numpy; visualization",
    "Machine Learning Engineer": "python; machine learning; deep learning; tensorflow; pytorch; mlops; docker; algorithms",
    "Data Analyst": "sql; excel; data analysis; visualization; statistics; tableau; python",
    "Data Engineer": "python; sql; etl; spark; aws; airflow; data pipelines; docker",
    "DevOps Engineer": "aws; docker; kubernetes; ci/cd; linux; git; automation; terraform; python",
    "Cloud Architect": "aws; azure; cloud; kubernetes; networking; security; terraform; automation",
    "Backend Developer": "java; python; sql; apis; git; data structures; docker; microservices",
    "Frontend Developer": "html; css; javascript; react; web design; git; typescript",
    "Full Stack Developer": "javascript; react; nodejs; sql; apis; html; css; git; docker",
    "Cybersecurity Analyst": "security; networking; linux; python; cryptography; penetration testing",
    "System Administrator": "linux; networking; bash; automation; security; cloud; virtualization",
    "AI Research Engineer": "python; deep learning; machine learning; algorithms; mathematics; pytorch; tensorflow",
}

# Roles shown when we cannot personalise (cold start fallback = "trending")
TRENDING_ROLES = ["Data Scientist", "DevOps Engineer", "Backend Developer"]

# Aliases so different names for the same skill map to ONE vocabulary term
ALIASES = {
    "ml": "machine_learning",
    "ai": "machine_learning",
    "dl": "deep_learning",
    "k8s": "kubernetes",
    "js": "javascript",
    "node": "nodejs",
    "node.js": "nodejs",
    "cloud_computing": "cloud",
    "frontend_development": "web_design",
    "web_development": "web_design",
    "cicd": "ci/cd",
    "ds": "data_structures",
    "dsa": "data_structures",
    "api": "apis",
    "amazon_web_services": "aws",
    "postgres": "sql",
    "mysql": "sql",
}


def normalize_skill(skill: str) -> str:
    """Lowercase, trim, join multi-word skills with '_' and resolve aliases."""
    s = skill.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(s, s)


def load_catalog() -> pd.DataFrame:
    """Load roles from raw_skills.csv if present, else use the built-in dataset."""
    if os.path.exists(CSV_PATH):
        try:
            raw = pd.read_csv(CSV_PATH)
            raw.columns = [c.strip().lower() for c in raw.columns]
            if {"role", "skills"}.issubset(raw.columns):
                raw = raw.dropna(subset=["role", "skills"])
                print(f"[+] Loaded {len(raw)} roles from {CSV_PATH}")
                return _build_catalog(zip(raw["role"], raw["skills"]))
            print(f"[!] {CSV_PATH} needs 'role' and 'skills' columns. Using built-in data.")
        except Exception as e:
            print(f"[!] Could not read {CSV_PATH} ({e}). Using built-in data.")
    print(f"[+] Using built-in dataset ({len(BUILT_IN_ROLES)} roles)")
    return _build_catalog(BUILT_IN_ROLES.items())


def _build_catalog(pairs) -> pd.DataFrame:
    rows = []
    for i, (role, skills) in enumerate(pairs, start=1):
        tokens = [normalize_skill(s) for s in str(skills).replace(",", ";").split(";") if s.strip()]
        rows.append({"item_id": i, "role": role, "skills": " ".join(tokens)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. Recommendation Engine Core
# ---------------------------------------------------------
def build_engine(df: pd.DataFrame):
    """Fit TF-IDF on the ITEMS ONLY so the user's input never changes the weights."""
    # Skills are already normalized tokens separated by spaces, so split on whitespace
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    item_vectors = vectorizer.fit_transform(df["skills"])
    return vectorizer, item_vectors


def get_recommendations(user_skills, df, vectorizer, item_vectors, top_n=TOP_N):
    """
    Step 1 Ingestion : user_skills (list of normalized skills)
    Step 2 Scoring   : cosine similarity vs every role
    Step 3 Sorting   : descending by score
    Step 4 Filtering : drop zero matches, keep Top-N
    Returns (results_df, unknown_skills). results_df is empty on cold start.
    """
    vocab = set(vectorizer.vocabulary_)
    known = [s for s in user_skills if s in vocab]
    unknown = [s for s in user_skills if s not in vocab]

    if not known:  # cold start: user vector would be all zeros
        return pd.DataFrame(), unknown

    user_vector = vectorizer.transform([" ".join(known)])           # transform only
    scores = cosine_similarity(user_vector, item_vectors).flatten()  # Step 2

    scored = df.copy()
    scored["similarity_score"] = scores
    scored = scored[scored["similarity_score"] > 0]                  # no 0% matches
    scored = scored.sort_values("similarity_score", ascending=False) # Step 3
    return scored.head(top_n), unknown                               # Step 4


def matched_skills(role_skills: str, user_skills) -> list:
    """Which of the user's skills does this role actually contain? (explainability)"""
    role_set = set(role_skills.split())
    return [s for s in user_skills if s in role_set]


# ---------------------------------------------------------
# 3. User Input (Ingestion)
# ---------------------------------------------------------
def get_user_skills() -> list:
    print(f"Enter at least {MIN_SKILLS} skills, separated by commas.")
    print("Example: Python, Cloud Computing, Automation\n")
    while True:
        raw = input("Your skills: ").strip()
        skills = []
        for part in raw.split(","):
            norm = normalize_skill(part) if part.strip() else ""
            if norm and norm not in skills:
                skills.append(norm)
        if len(skills) >= MIN_SKILLS:
            return skills
        print(f"[!] Please provide at least {MIN_SKILLS} different skills "
              f"(you gave {len(skills)}).\n")


# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
def main():
    print("==================================================")
    print("     PROJECT 3: TECH STACK RECOMMENDER (AI)       ")
    print("==================================================\n")

    df = load_catalog()
    vectorizer, item_vectors = build_engine(df)
    print(f"[+] Vocabulary size: {len(vectorizer.vocabulary_)} unique skills\n")

    user_skills = get_user_skills()
    print(f"\n[+] Normalized profile: {user_skills}")
    print("[+] Scoring with TF-IDF & Cosine Similarity...\n")

    results, unknown = get_recommendations(user_skills, df, vectorizer, item_vectors)

    if unknown:
        print(f"[!] Skills not in our dataset (ignored): {unknown}\n")

    if results.empty:
        # Cold-start bypass: trending fallback
        print("[!] None of your skills matched our dataset.")
        print("--- TRENDING CAREER PATHS (fallback) ---")
        for i, role in enumerate(TRENDING_ROLES, start=1):
            print(f"{i}. {role}")
        return

    print(f"--- TOP {len(results)} CAREER PATHS FOR YOU ---")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        pct = row["similarity_score"] * 100
        why = ", ".join(matched_skills(row["skills"], user_skills))
        print(f"{rank}. {row['role']:<28} Match: {pct:5.2f}%   (matched: {why})")


if __name__ == "__main__":
    main()