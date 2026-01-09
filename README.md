🏏 Match Outcome Probability Prediction Using Machine Learning (IPL Case Study)
<p align="left"> <img src="https://img.shields.io/badge/Project-ML%20Web%20App-blue.svg" /> <img src="https://img.shields.io/badge/Framework-Flask-green.svg" /> <img src="https://img.shields.io/badge/Language-Python%203.10-orange.svg" /> <img src="https://img.shields.io/badge/Status-Active-success.svg" /> </p>
📌 Overview

The IPL Win Probability Predictor is a Machine Learning–powered web application that predicts the winning chances of two IPL teams in real-time based on multiple match parameters, including:

Score

Overs completed

Wickets fallen

Target

Host city

Toss winner & toss decision

This project was developed as part of my AI/ML Internship at InternPe, focusing on real-world model deployment with an interactive and clean user interface.

🚀 Key Features

✔ Real-time win probability predictions
✔ Clean & interactive web UI
✔ Flask-based backend
✔ Trained ML model with encoded input data
✔ Dynamic probability bars
✔ Lightweight and fast — ideal for local or cloud deployment
✔ Great template for sports analytics or live prediction systems

🧠 Machine Learning Approach

Model: Random Forest Classifier

Training Data: IPL historical match outcomes

Feature Engineering:

Batting & bowling teams

Toss winner & decision

City/venue

Additional Calculations:

Run rate vs required rate

Wickets impact

Overs pressure factor

Output: Win probability distribution between the two teams

🖥️ Demo Output
🔹 Prediction Example
<img width="1366" height="768" alt="2025-12-09 (8)" src="https://github.com/user-attachments/assets/e8d2b977-a54a-48a3-917c-87ad423f2f6e" />

🔹 Another Example
<img width="1366" height="768" alt="2025-12-09 (7)" src="https://github.com/user-attachments/assets/b2b37deb-2ec1-45d4-a677-c3c34fb75a8f" />

🔹 Clean Input UI
<img width="1366" height="768" alt="2025-12-09 (8)" src="https://github.com/user-attachments/assets/d83f86f7-a808-41cd-909b-6f5a57036d0f" />

⚙️ Tech Stack
Backend

Python

Flask

scikit-learn

pandas

joblib

Frontend

HTML

CSS

Bootstrap (optional)

Model Files

model.pkl

encoders.pkl

📂 Project Structure
ipl-win-probability-predictor/
│
├── app.py
├── train_model.py
│
├── models/
│   ├── model.pkl
│   └── encoders.pkl
│
├── data/
│   └── matches.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md

🔧 Installation & Running Locally
1️⃣ Clone the repository
git clone https://github.com/yourusername/ipl-win-probability-predictor.git
cd ipl-win-probability-predictor

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional if model is included)
python train_model.py

4️⃣ Run the Flask app
python app.py

🌐 Deployment

This project can be easily deployed on:

Render

Railway

Heroku

AWS EC2

PythonAnywhere

Ensure the following files are included in your deployment package:
✔ model.pkl
✔ encoders.pkl
✔ requirements.txt

🏆 Internship

This project was created as part of my AI/ML Internship at InternPe, focusing on:

Real-time prediction systems

Flask-based deployment

Model integration

Clean user interface design

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you'd like to improve.

📬 Contact

Developer: M V Karthikeya
📧 Email: mvkarthikeya2005@gmail.com

🔗 LinkedIn: www.linkedin.com/in/mv-karthikeya-b26a2131b

📜 License

This project is licensed under the MIT License.
