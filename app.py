from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'balaji-1805'

# Load trained models
with open('models/anxiety_model.pkl', 'rb') as f:
    model_anxiety = pickle.load(f)
with open('models/depression_model.pkl', 'rb') as f:
    model_depression = pickle.load(f)

# CSV file path
USER_CSV = 'users.csv'

# Ensure CSV exists
if not os.path.exists(USER_CSV):
    df_init = pd.DataFrame(columns=['username', 'email', 'password'])
    df_init.to_csv(USER_CSV, index=False)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password'].strip()

        users_df = pd.read_csv(USER_CSV)
        users_df['email'] = users_df['email'].str.strip().str.lower()

        if email in users_df['email'].values:
            flash('Email already exists. Please use a different email.')
        else:
            new_user = pd.DataFrame([[name, email, password]], columns=['username', 'email', 'password'])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USER_CSV, index=False)
            flash('Account created successfully! Please login.')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        users_df = pd.read_csv(USER_CSV)
        users_df['email'] = users_df['email'].str.strip().str.lower()
        users_df['password'] = users_df['password'].astype(str).str.strip()

        match = users_df[
            (users_df['email'] == email) & 
            (users_df['password'] == password)
        ]

        if not match.empty:
            flash('Login successful!')
            return redirect(url_for('form'))  # Redirect to prediction form
        else:
            error = "Invalid email or password."
            return render_template('login.html', error=error)

    return render_template('login.html')
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            user_data = {k: float(request.form[k]) for k in request.form}
            input_data = pd.DataFrame([user_data])

            anxiety_prob = model_anxiety.predict_proba(input_data)[0][1]
            depression_prob = model_depression.predict_proba(input_data)[0][1]

            anxiety_result = "High Risk" if anxiety_prob > 0.5 else "Low Risk"
            depression_result = "High Risk" if depression_prob > 0.5 else "Low Risk"

            # Redirect to results with values
            return redirect(url_for('results', anxiety=anxiety_result, depression=depression_result))

        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for('form'))

    return render_template('form.html')

@app.route('/results', methods=['GET'])
def results():
    anxiety = request.args.get('anxiety', 'No data')
    depression = request.args.get('depression', 'No data')

    # Define recommendations based on risk
    if anxiety == 'High Risk' or depression == 'High Risk':
        recommendations = [
            "Practice daily mindfulness or meditation (10–15 minutes).",
            "Maintain a consistent sleep schedule (7–8 hours/night).",
            "Exercise regularly – even a 20-min walk helps.",
            "Limit screen time and avoid excessive social media.",
            "Write in a gratitude or reflective journal daily.",
            "Talk to a trusted friend or family member regularly.",
            "Spend time on hobbies or creative activities you enjoy."
        ]
    else:
        recommendations = [
            "Keep practicing positive habits like exercise and mindfulness.",
            "Maintain social connections with friends and loved ones.",
            "Balance your daily routine between work and relaxation.",
            "Stay consistent with good sleep and healthy food choices.",
            "Continue activities that help you feel calm and focused."
        ]

    return render_template(
        'results.html',
        anxiety_risk=anxiety,
        depression_risk=depression,
        recommendations=recommendations
    )

if __name__ == '__main__':
    app.run(debug=True)
