"""
Student Performance Prediction - Flask Application
Main application file with routes and logic
"""

from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from database import init_db, save_prediction, get_all_predictions
from model import StudentPerformanceModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize database
init_db()

# Initialize ML model
ml_model = StudentPerformanceModel()
ml_model.train_model()

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        student_name = request.form.get('student_name', 'Anonymous')
        attendance = float(request.form.get('attendance'))
        study_hours = float(request.form.get('study_hours'))
        previous_marks = float(request.form.get('previous_marks'))
        
        # Validate inputs
        if not (0 <= attendance <= 100):
            return render_template('index.html', error='Attendance must be between 0 and 100')
        if not (0 <= study_hours <= 24):
            return render_template('index.html', error='Study hours must be between 0 and 24')
        if not (0 <= previous_marks <= 100):
            return render_template('index.html', error='Previous marks must be between 0 and 100')
        
        # Make predictions using Random Forest (better accuracy)
        predicted_score, pass_fail, model_used = ml_model.predict(
            attendance, study_hours, previous_marks, use_random_forest=True
        )
        
        # Save to database
        prediction_id = save_prediction(
            student_name=student_name,
            attendance=attendance,
            study_hours=study_hours,
            previous_marks=previous_marks,
            predicted_score=predicted_score,
            pass_fail=pass_fail
        )
        
        # Generate visualization
        chart_path = generate_chart(attendance, study_hours, previous_marks, predicted_score)
        
        # Get feature importance for display
        feature_importance = ml_model.get_feature_importance()
        
        return render_template('result.html',
                             student_name=student_name,
                             attendance=attendance,
                             study_hours=study_hours,
                             previous_marks=previous_marks,
                             predicted_score=predicted_score,
                             pass_fail=pass_fail,
                             model_used=model_used,
                             chart_path=chart_path,
                             feature_importance=feature_importance)
    
    except ValueError:
        return render_template('index.html', error='Please enter valid numeric values')
    except Exception as e:
        return render_template('index.html', error=f'An error occurred: {str(e)}')

@app.route('/history')
def history():
    """Display prediction history"""
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)

def generate_chart(attendance, study_hours, previous_marks, predicted_score):
    """Generate visualization chart"""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Chart 1: Input Parameters
        categories = ['Attendance\n(%)', 'Study Hours\n(per day)', 'Previous\nMarks']
        values = [attendance, study_hours * 4.17, previous_marks]  # Scale study hours for visualization
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        axes[0].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Input Parameters', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[0].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Predicted Score
        pass_threshold = 40
        colors_score = ['#4CAF50' if predicted_score >= pass_threshold else '#F44336']
        
        axes[1].bar(['Predicted Score'], [predicted_score], color=colors_score, alpha=0.7, edgecolor='black', width=0.5)
        axes[1].axhline(y=pass_threshold, color='red', linestyle='--', linewidth=2, label=f'Pass Threshold ({pass_threshold})')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Predicted Performance', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].legend()
        
        # Add value label
        axes[1].text(0, predicted_score + 2, f'{predicted_score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f'chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        chart_path = os.path.join('static', 'charts', chart_filename)
        
        # Create charts directory if it doesn't exist
        os.makedirs(os.path.join('static', 'charts'), exist_ok=True)
        
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    except Exception as e:
                print(f"Error generating chart: {e}")
                return None

        if __name__ == '__main__':
                port = int(os.environ.get('PORT', 7860))
                app.run(debug=False, host='0.0.0.0', port=port)
            
