from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import mysql.connector
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages


# Database connection setup
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='balaji@0000',
        database='StudentMentalHealthDB'
    )


# Load trained Logistic Regression models
with open('models/anxiety_model.pkl', 'rb') as f:
    model_anxiety = pickle.load(f)

with open('models/depression_model.pkl', 'rb') as f:
    model_depression = pickle.load(f)

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (name, email, password))
            conn.commit()
            flash('Account created successfully! Please login.')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists. Please use a different email.')
        finally:
            cursor.close()
            conn.close()

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            flash('Login successful!')
            return redirect(url_for('form'))
        else:
            flash('Invalid email or password.')

    return render_template('login.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            # Collect user inputs
            user_data = {k: float(request.form[k]) for k in request.form}
            input_data = pd.DataFrame([user_data])

            # Predictions
            anxiety_prob = model_anxiety.predict_proba(input_data)[0][1]
            depression_prob = model_depression.predict_proba(input_data)[0][1]

            anxiety_result = "High Risk of Anxiety" if anxiety_prob > 0.5 else "Low Risk of Anxiety"
            depression_result = "High Risk of Depression" if depression_prob > 0.5 else "Low Risk of Depression"

            # Redirect to results page with query parameters
            return redirect(url_for('results', anxiety=anxiety_result, depression=depression_result))

        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for('form'))

    return render_template('form.html')

@app.route('/results', methods=['GET'])
def results():
    anxiety = request.args.get('anxiety', 'No data')
    depression = request.args.get('depression', 'No data')
    return render_template('results.html', anxiety_risk=anxiety, depression_risk=depression)


if __name__ == '__main__':
    app.run(debug=True)
