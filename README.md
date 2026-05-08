 # 🎓 Student Performance Prediction & Analytics Suite

A comprehensive machine learning system designed to predict student performance (numerical score and pass/fail classification) based on behavioral and academic features. Built with a robust Python/Flask backend and an interactive web interface.

---

## 🚀 Key Features
* **Multi-Algorithm Model Training:** Implements and compares multiple predictive models:
  * **Regression (Score Prediction):** Linear Regression vs. Random Forest Regressor.
  * **Classification (Pass/Fail):** Logistic Regression vs. Random Forest Classifier.
* **Interactive Flask Web UI:** A clean, responsive form where users can input student metrics and receive instant real-time predictions.
* **Data Visualization:** Generates dynamic **Matplotlib** charts comparing the student's metrics to class averages and passing thresholds.
* **Persistent Database Logging:** Automatically saves all input parameters and predicted metrics to an SQLite database (`students.db`) for long-term tracking.
* **Historical Performance Dashboard:** A dedicated `/history` panel to view, search, and analyze all past performance predictions.

---

## 🛠️ Technology Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (including `StandardScaler` pipeline normalization)
* **Web Framework:** Flask (Backend) + HTML5 / CSS3 / Bootstrap (Frontend)
* **Data Science:** Pandas, NumPy, Matplotlib
* **Database:** SQLite3

---

## 📦 Directory Structure
```text
Student-Performance-Prediction/
├── app.py              # Flask Web Application & Route Handlers
├── database.py         # SQLite Database Schema & Session Controller
├── model.py            # Machine Learning Model Pipeline (Train/Predict)
├── students.db         # Persistent SQLite Database
└── README.md           # Project Documentation
```

---

## 🧠 Model & Feature Architecture

### Feature Input Matrix:
1. **Attendance (%):** Percentage of lectures attended.
2. **Study Hours per Day:** Average hours spent studying daily.
3. **Previous Marks (%):** Prior academic performance.

### Preprocessing & Pipeline:
* Inputs are scaled using a fitted **StandardScaler** pipeline to ensure optimal weight distribution across regression and classification models.
* **Random Forest Regressor** is utilized as the default production model for score prediction due to its superior non-linear fitting capabilities.
* **Pass/Fail Threshold:** Set at **40%**. If the predicted score is >= 40, the student is classified as "Pass"; otherwise, "Fail".

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bilalahmed251/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install flask scikit-learn pandas numpy matplotlib
   ```

3. **Train the models & initialize the database:**
   ```bash
   python model.py
   python database.py
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```
   * Open your browser and navigate to `http://127.0.0.1:5000` to interact with the engine.

---

## 📊 Database Schema (`students.db`)
Predictions are stored in a structured `predictions` table with the following schema:
* `id` (Primary Key, Auto-increment)
* `attendance` (Float)
* `study_hours` (Float)
* `previous_marks` (Float)
* `predicted_score` (Float)
* `status` (String - "Pass" or "Fail")
* `timestamp` (DateTime)

---

## 🌟 Recruiter Review Highlights
* **Full-Stack ML Engineering:** Showcases integration of machine learning pipelines with production web frameworks (Flask) and relational databases (SQLite).
* **Robust Input Validation:** Web form prevents out-of-bound inputs (e.g. attendance > 100%).
* **Data Visualization:** Dynamic charting shows a deep understanding of analytical reporting for end-users.
