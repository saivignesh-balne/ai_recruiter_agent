import os
import random

import docx
import PyPDF2
import spacy
import together  # Add this import
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from werkzeug.utils import secure_filename

# Configure Together API (or OpenAI as fallback)
TOGETHER_API_KEY = "1afbc1f7485902e08288d917082a3881378a6b61cf04b49173ec7bee5f239ca0"
together.api_key = TOGETHER_API_KEY

# Load spaCy model (run `python -m spacy download en_core_web_lg` first)
nlp = spacy.load("en_core_web_lg")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Update your JOB_DB to include preferred skills
JOB_DB = {
    'software_engineer': {
        'title': 'Software Engineer',
        'description': 'Looking for a software engineer with 3+ years experience in Python, JavaScript, and cloud technologies. Strong problem-solving skills required. Experience with Docker, Kubernetes, and CI/CD pipelines preferred.',
        'keywords': ['python', 'javascript', 'cloud', 'algorithms', 'problem solving'],
        'preferred': ['docker', 'kubernetes', 'ci/cd', 'aws', 'azure']
    },
    'data_scientist': {
        'title': 'Data Scientist',
        'description': 'Seeking data scientist with expertise in machine learning, statistical analysis, and data visualization. Experience with Python and SQL required. Knowledge of deep learning frameworks and big data tools preferred.',
        'keywords': ['machine learning', 'statistics', 'python', 'sql', 'data visualization'],
        'preferred': ['tensorflow', 'pytorch', 'spark', 'hadoop']
    }
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ' '.join([page.extract_text() for page in reader.pages])
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        text = ' '.join([para.text for para in doc.paragraphs])
    return text

def enhanced_score_resume(resume_text, job_role):
    job_data = JOB_DB[job_role]
    job_desc = job_data['description']
    required_skills = job_data['keywords']
    preferred_skills = job_data.get('preferred', [])
    
    # TF-IDF Similarity (40% weight)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Keyword Matching (30% weight)
    resume_lower = resume_text.lower()
    matched_required = [kw for kw in required_skills if kw in resume_lower]
    matched_preferred = [kw for kw in preferred_skills if kw in resume_lower]
    keyword_score = (
        (len(matched_required) / len(required_skills) * 0.7) +
        ((len(matched_preferred) / len(preferred_skills) * 0.3) if preferred_skills else 0)
    )
    
    # Semantic Similarity using spaCy (30% weight)
    doc1 = nlp(resume_text)
    doc2 = nlp(job_desc)
    semantic_similarity = doc1.similarity(doc2)
    
    # Combined score
    final_score = (
        0.5 * tfidf_similarity +
        0.5 * keyword_score +
        0.5 * semantic_similarity
    ) * 100  # Convert to percentage
    
    # Add small random variation to avoid identical scores
    final_score = min(100, max(0, final_score + random.uniform(-2, 2)))
    
    return round(final_score, 1)

def score_resume(resume_text, job_role):
    # Deprecated: use enhanced_score_resume instead
    return enhanced_score_resume(resume_text, job_role)

def generate_ai_questions(resume_text, job_role):
    try:
        # Try Together API first
        prompt = f"""
        Generate 5 technical interview questions based on this resume for a {JOB_DB[job_role]['title']} position.
        Focus on their specific skills and experiences mentioned in the resume.
        
        Resume:
        {resume_text[:3000]}  # Limit to first 3000 chars
        
        Return the questions as a numbered list.
        """
        
        response = together.Complete.create(
            prompt=prompt,
            model="togethercomputer/llama-2-70b-chat",
            max_tokens=500,
            temperature=0.7,
            top_k=50,
            top_p=0.7
        )
        
        questions = response['output']['choices'][0]['text']
        return [q.strip() for q in questions.split('\n') if q.strip() and q[0].isdigit()]
    
    except Exception as e:
        print(f"Together API failed, using fallback: {e}")
        # Fallback to simpler generation
        doc = nlp(resume_text)
        
        # Extract nouns and verbs as topics
        topics = [chunk.text for chunk in doc.noun_chunks][:5]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"][:3]
        
        questions = [
            f"Can you describe your experience with {topics[0]}?",
            f"What was your most challenging project involving {topics[1]}?",
            f"How do you approach {verbs[0]} when working with {topics[2]}?",
            f"What metrics do you use to measure success with {topics[3]}?",
            f"Tell me about a time you had to troubleshoot an issue with {topics[4]}."
        ]
        
        return questions[:3]  # Return first 3 questions

def generate_interview_questions(resume_text):
    # This would be enhanced with proper NLP in production
    questions = [
        "Can you walk us through your experience with the technologies mentioned in your resume?",
        "What was your most challenging project and how did you approach it?",
        "How do you stay updated with the latest developments in your field?",
        "Describe a time when you had to solve a difficult technical problem.",
        "What are your strengths and areas for improvement?"
    ]
    return questions[:3]  # Return first 3 questions for demo

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start-process')
def start_process():
    return render_template('form.html')

@app.route('/submit-form', methods=['POST'])
def submit_form():
    if 'resume' not in request.files:
        return redirect(request.url)
    
    file = request.files['resume']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process form data
        name = request.form['name']
        job_role = request.form['job_role']
        
        # Extract and score resume
        resume_text = extract_text_from_file(filepath)
        score = enhanced_score_resume(resume_text, job_role)
        
        # Store in session
        session['candidate_data'] = {
            'name': name,
            'job_role': job_role,
            'resume_score': score,
            'resume_path': filepath,
            'resume_text': resume_text
        }
        
        if score >= 50:  # Threshold for demo
            return redirect(url_for('interview'))
        else:
            return render_template('rejected.html', score=score)
    
    return redirect(request.url)

@app.route('/interview')
def interview():
    candidate_data = session.get('candidate_data')
    if not candidate_data:
        return redirect(url_for('start_process'))
    
    # Use AI-powered question generation
    questions = generate_ai_questions(candidate_data['resume_text'], candidate_data['job_role'])
    session['interview_questions'] = questions
    session['current_question'] = 0
    session['interview_answers'] = []
    
    return render_template('interview.html', 
                         name=candidate_data['name'],
                         job_role=JOB_DB[candidate_data['job_role']]['title'],
                         question=questions[0],
                         question_num=1,
                         total_questions=len(questions))

@app.route('/process-answer', methods=['POST'])
def process_answer():
    answer_text = request.form['answer']
    candidate_data = session.get('candidate_data')
    
    # Store answer with sentiment analysis
    blob = TextBlob(answer_text)
    sentiment = blob.sentiment.polarity  # -1 to 1
    
    session['interview_answers'].append({
        'text': answer_text,
        'sentiment': sentiment,
        'length': len(answer_text.split())
    })
    
    # Move to next question or finish
    current_q = session['current_question'] + 1
    questions = session['interview_questions']
    
    if current_q < len(questions):
        session['current_question'] = current_q
        return render_template('interview.html', 
                            name=candidate_data['name'],
                            job_role=JOB_DB[candidate_data['job_role']]['title'],
                            question=questions[current_q],
                            question_num=current_q+1,
                            total_questions=len(questions))
    else:
        # Enhanced scoring
        interview_score = score_interview(session['interview_answers'])
        resume_score = candidate_data['resume_score']
        final_score = 0.4 * resume_score + 0.6 * interview_score
        
        passed = final_score >= 70
        session.clear()
        
        return render_template('result.html', 
                            resume_score=resume_score,
                            interview_score=interview_score,
                            final_score=final_score,
                            passed=passed)

def score_interview(answers):
    """Score interview based on answer quality"""
    total_score = 0
    
    for answer in answers:
        # Score based on length (20%)
        length_score = min(1, answer['length'] / 50) * 20
        
        # Score based on sentiment (30%)
        sentiment_score = (answer['sentiment'] + 1) * 15  # Convert -1 to 1 range to 0-30
        
        # Score based on keywords (50%)
        keywords = JOB_DB[session['candidate_data']['job_role']]['keywords']
        keyword_score = sum(1 for kw in keywords if kw in answer['text'].lower()) / len(keywords) * 50
        
        total_score += length_score + sentiment_score + keyword_score
    
    return total_score / len(answers)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)