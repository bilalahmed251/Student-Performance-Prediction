"""
Database operations for Student Performance Prediction
Uses SQLite to store student inputs and predictions
"""

import sqlite3
from datetime import datetime
import os

DATABASE_NAME = 'students.db'

def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            attendance REAL NOT NULL,
            study_hours REAL NOT NULL,
            previous_marks REAL NOT NULL,
            predicted_score REAL NOT NULL,
            pass_fail TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def save_prediction(student_name, attendance, study_hours, previous_marks, predicted_score, pass_fail):
    """
    Save a prediction to the database
    
    Args:
        student_name: Name of the student
        attendance: Attendance percentage
        study_hours: Study hours per day
        previous_marks: Previous exam marks
        predicted_score: Predicted final score
        pass_fail: Pass or Fail result
        
    Returns:
        prediction_id: ID of the saved prediction
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (student_name, attendance, study_hours, previous_marks, predicted_score, pass_fail)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (student_name, attendance, study_hours, previous_marks, predicted_score, pass_fail))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id

def get_all_predictions():
    """
    Retrieve all predictions from the database
    
    Returns:
        List of prediction records
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM predictions
        ORDER BY timestamp DESC
    ''')
    
    predictions = cursor.fetchall()
    conn.close()
    
    return predictions

def get_prediction_by_id(prediction_id):
    """
    Retrieve a specific prediction by ID
    
    Args:
        prediction_id: ID of the prediction
        
    Returns:
        Prediction record or None
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
    prediction = cursor.fetchone()
    
    conn.close()
    return prediction

def get_statistics():
    """
    Get statistics from all predictions
    
    Returns:
        Dictionary with statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total_predictions,
            AVG(predicted_score) as avg_score,
            MAX(predicted_score) as max_score,
            MIN(predicted_score) as min_score,
            SUM(CASE WHEN pass_fail = 'Pass' THEN 1 ELSE 0 END) as total_pass,
            SUM(CASE WHEN pass_fail = 'Fail' THEN 1 ELSE 0 END) as total_fail
        FROM predictions
    ''')
    
    stats = cursor.fetchone()
    conn.close()
    
    return dict(stats) if stats else {}

def delete_all_predictions():
    """Delete all predictions from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()

# Test database operations
if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Test saving a prediction
    prediction_id = save_prediction(
        student_name="Test Student",
        attendance=85.0,
        study_hours=6.0,
        previous_marks=75.0,
        predicted_score=78.5,
        pass_fail="Pass"
    )
    print(f"Saved prediction with ID: {prediction_id}")
    
    # Test retrieving predictions
    predictions = get_all_predictions()
    print(f"\nTotal predictions: {len(predictions)}")
    
    # Test statistics
    stats = get_statistics()
    print(f"\nStatistics: {stats}")
