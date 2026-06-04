import fitz
import sqlite3

# Open PDF
pdf = fitz.open("Rajitha_Resume.pdf")

resume_text = ""

for page in pdf:
    resume_text += page.get_text()

print("Resume Content:")
print(resume_text)

# Simple NLP Skill Matching
required_skills = [
    "Python",
    "SQL",
    "Machine Learning",
    "Pandas",
    "NumPy"
]

score = 0

for skill in required_skills:
    if skill.lower() in resume_text.lower():
        score += 1

print("\nResume Score:", score)

# Store in Database
connection = sqlite3.connect("resumes.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
score INTEGER
)
""")

cursor.execute(
    "INSERT INTO candidates(name, score) VALUES (?, ?)",
    ("Rajitha", score)
)

connection.commit()
connection.close()

print("Result stored successfully.")