from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load dataset
dataset_path = "NEW_dataset.csv"  # Replace with your new CSV file name
student_data = pd.read_csv(dataset_path)

# Strip column names to avoid leading/trailing spaces
student_data.columns = student_data.columns.str.strip()

# Set OpenAI API Key
openai.api_key = "your-api-key"

# Function to call GPT-4 Chat API
def get_gpt4_recommendation(student_id, electives, student_marks, student_name):
    messages = [
        {"role": "system", "content": "You are an expert in academic counseling."},
        {
            "role": "user",
            "content": (
                f"Student Name: {student_name}. "
                f"The student with ID {student_id} has the following scores: {student_marks}. "
                f"The available elective options are: {', '.join(electives)}. "
                f"Based on the student's performance, recommend the best elective."
            ),
        },
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error calling GPT-4: {str(e)}"

# API endpoint for recommendation
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    student_id = data.get('student_id')
    electives = data.get('elective_options')

    if not student_id or not electives:
        return jsonify({"error": "Invalid input"}), 400

    # Fetch student details and marks from dataset
    student_row = student_data[student_data['student_id'] == student_id]
    if student_row.empty:
        return jsonify({"error": "Student ID not found"}), 404

    student_name = student_row['Name'].values[0]
    elective_columns = [
        'Cloud Computing', 
        'Data Analysis and Visualization', 
        'Statistical Foundation For Data Science', 
        'Applied Artificial Intelligence', 
        'Applied Machine Learning', 
        'Human Computer Interaction'
    ]
    student_marks = student_row[elective_columns].iloc[0].to_dict()

    # Get GPT-4 recommendation
    recommendation = get_gpt4_recommendation(student_id, electives, student_marks, student_name)
    return jsonify({"recommended_elective": recommendation})

if __name__ == '__main__':
    app.run(debug=True)