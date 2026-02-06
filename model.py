"""
Machine Learning Model for Student Performance Prediction
Uses multiple models: Linear Regression, Logistic Regression, Random Forest
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
import os

class StudentPerformanceModel:
    def __init__(self):
        # Linear models
        self.linear_score_model = None  # Linear Regression for score prediction
        self.logistic_pass_fail_model = None  # Logistic Regression for pass/fail
        
        # Random Forest models (Better performance)
        self.rf_score_model = None  # Random Forest Regressor for score prediction
        self.rf_pass_fail_model = None  # Random Forest Classifier for pass/fail
        
        self.scaler = StandardScaler()
        self.pass_threshold = 40  # Marks below 40 is fail
        
        # Store feature importance
        self.feature_names = ['Attendance (%)', 'Study Hours/Day', 'Previous Marks (%)']
        self.feature_importance_score = None
        self.feature_importance_pass_fail = None
        
    def train_model(self):
        """
        Train all ML models with synthetic training data
        In a real scenario, you would load actual historical data
        """
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 500
        
        # Features: Attendance, Study Hours, Previous Marks
        attendance = np.random.uniform(30, 100, n_samples)
        study_hours = np.random.uniform(0, 10, n_samples)
        previous_marks = np.random.uniform(20, 100, n_samples)
        
        # Create feature matrix
        X = np.column_stack([attendance, study_hours, previous_marks])
        
        # Generate target scores with realistic relationships
        # Score depends on: 30% attendance, 25% study hours, 45% previous marks
        noise = np.random.normal(0, 5, n_samples)
        y_score = (
            0.30 * attendance +
            0.25 * study_hours * 10 +  # Scale study hours
            0.45 * previous_marks +
            noise
        )
        
        # Clip scores to 0-100 range
        y_score = np.clip(y_score, 0, 100)
        
        # Create pass/fail labels (1 = Pass, 0 = Fail)
        y_pass_fail = (y_score >= self.pass_threshold).astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training models...")
        print("=" * 60)
        
        # ===== TRAIN LINEAR REGRESSION (Score Prediction) =====
        self.linear_score_model = LinearRegression()
        self.linear_score_model.fit(X_scaled, y_score)
        linear_r2 = self.linear_score_model.score(X_scaled, y_score)
        print(f"✓ Linear Regression (Score) - R² Score: {linear_r2:.4f}")
        
        # ===== TRAIN RANDOM FOREST REGRESSOR (Score Prediction) =====
        self.rf_score_model = RandomForestRegressor(
            n_estimators=100,  # Number of trees
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.rf_score_model.fit(X_scaled, y_score)
        rf_r2 = self.rf_score_model.score(X_scaled, y_score)
        self.feature_importance_score = self.rf_score_model.feature_importances_
        print(f"✓ Random Forest Regressor (Score) - R² Score: {rf_r2:.4f}")
        
        # ===== TRAIN LOGISTIC REGRESSION (Pass/Fail) =====
        self.logistic_pass_fail_model = LogisticRegression(random_state=42, max_iter=1000)
        self.logistic_pass_fail_model.fit(X_scaled, y_pass_fail)
        logistic_acc = self.logistic_pass_fail_model.score(X_scaled, y_pass_fail)
        print(f"✓ Logistic Regression (Pass/Fail) - Accuracy: {logistic_acc:.4f}")
        
        # ===== TRAIN RANDOM FOREST CLASSIFIER (Pass/Fail) =====
        self.rf_pass_fail_model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.rf_pass_fail_model.fit(X_scaled, y_pass_fail)
        rf_acc = self.rf_pass_fail_model.score(X_scaled, y_pass_fail)
        self.feature_importance_pass_fail = self.rf_pass_fail_model.feature_importances_
        print(f"✓ Random Forest Classifier (Pass/Fail) - Accuracy: {rf_acc:.4f}")
        
        print("=" * 60)
        print("\n📊 Feature Importance (Random Forest - Score Prediction):")
        for name, importance in zip(self.feature_names, self.feature_importance_score):
            print(f"   {name}: {importance:.4f} ({importance*100:.2f}%)")
        
        print("\n📊 Feature Importance (Random Forest - Pass/Fail):")
        for name, importance in zip(self.feature_names, self.feature_importance_pass_fail):
            print(f"   {name}: {importance:.4f} ({importance*100:.2f}%)")
        
        # Save models
        self.save_models()
        
        print("\n✅ All models trained and saved successfully!")
        
    def predict(self, attendance, study_hours, previous_marks, use_random_forest=True):
        """
        Make predictions for a student
        
        Args:
            attendance: Attendance percentage (0-100)
            study_hours: Study hours per day (0-24)
            previous_marks: Previous exam marks (0-100)
            use_random_forest: If True, use Random Forest models (default), else use Linear models
            
        Returns:
            predicted_score: Predicted final score
            pass_fail: "Pass" or "Fail"
            model_used: Name of the model used
        """
        # Load models if not already loaded
        if self.rf_score_model is None:
            self.load_models()
        
        # Prepare input
        X = np.array([[attendance, study_hours, previous_marks]])
        X_scaled = self.scaler.transform(X)
        
        if use_random_forest:
            # Use Random Forest models (Better accuracy)
            predicted_score = self.rf_score_model.predict(X_scaled)[0]
            pass_fail_prediction = self.rf_pass_fail_model.predict(X_scaled)[0]
            model_used = "Random Forest"
        else:
            # Use Linear models
            predicted_score = self.linear_score_model.predict(X_scaled)[0]
            pass_fail_prediction = self.logistic_pass_fail_model.predict(X_scaled)[0]
            model_used = "Linear/Logistic Regression"
        
        # Ensure score is in valid range
        predicted_score = np.clip(predicted_score, 0, 100)
        
        # Convert pass/fail prediction to text
        pass_fail = "Pass" if pass_fail_prediction == 1 else "Fail"
        
        # Double-check with score threshold for consistency
        if predicted_score >= self.pass_threshold:
            pass_fail = "Pass"
        else:
            pass_fail = "Fail"
        
        return round(predicted_score, 2), pass_fail, model_used
    
    def predict_all_models(self, attendance, study_hours, previous_marks):
        """
        Get predictions from all models for comparison
        
        Returns:
            Dictionary with predictions from all models
        """
        X = np.array([[attendance, study_hours, previous_marks]])
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        # Linear Regression
        lr_score = np.clip(self.linear_score_model.predict(X_scaled)[0], 0, 100)
        results['Linear Regression'] = {
            'score': round(lr_score, 2),
            'pass_fail': 'Pass' if lr_score >= self.pass_threshold else 'Fail'
        }
        
        # Random Forest Regressor
        rf_score = np.clip(self.rf_score_model.predict(X_scaled)[0], 0, 100)
        results['Random Forest'] = {
            'score': round(rf_score, 2),
            'pass_fail': 'Pass' if rf_score >= self.pass_threshold else 'Fail'
        }
        
        # Logistic Regression
        log_pred = self.logistic_pass_fail_model.predict(X_scaled)[0]
        results['Logistic Regression'] = {
            'prediction': 'Pass' if log_pred == 1 else 'Fail'
        }
        
        # Random Forest Classifier
        rf_pred = self.rf_pass_fail_model.predict(X_scaled)[0]
        results['Random Forest Classifier'] = {
            'prediction': 'Pass' if rf_pred == 1 else 'Fail'
        }
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest models"""
        return {
            'score_prediction': dict(zip(self.feature_names, self.feature_importance_score)),
            'pass_fail_prediction': dict(zip(self.feature_names, self.feature_importance_pass_fail))
        }
    
    def save_models(self):
        """Save all trained models to disk"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save Linear models
        with open(os.path.join(models_dir, 'linear_score_model.pkl'), 'wb') as f:
            pickle.dump(self.linear_score_model, f)
        
        with open(os.path.join(models_dir, 'logistic_pass_fail_model.pkl'), 'wb') as f:
            pickle.dump(self.logistic_pass_fail_model, f)
        
        # Save Random Forest models
        with open(os.path.join(models_dir, 'rf_score_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_score_model, f)
        
        with open(os.path.join(models_dir, 'rf_pass_fail_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_pass_fail_model, f)
        
        # Save scaler and feature importance
        with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(models_dir, 'feature_importance.pkl'), 'wb') as f:
            pickle.dump({
                'score': self.feature_importance_score,
                'pass_fail': self.feature_importance_pass_fail,
                'names': self.feature_names
            }, f)
    
    def load_models(self):
        """Load all trained models from disk"""
        models_dir = 'models'
        
        try:
            # Load Linear models
            with open(os.path.join(models_dir, 'linear_score_model.pkl'), 'rb') as f:
                self.linear_score_model = pickle.load(f)
            
            with open(os.path.join(models_dir, 'logistic_pass_fail_model.pkl'), 'rb') as f:
                self.logistic_pass_fail_model = pickle.load(f)
            
            # Load Random Forest models
            with open(os.path.join(models_dir, 'rf_score_model.pkl'), 'rb') as f:
                self.rf_score_model = pickle.load(f)
            
            with open(os.path.join(models_dir, 'rf_pass_fail_model.pkl'), 'rb') as f:
                self.rf_pass_fail_model = pickle.load(f)
            
            # Load scaler
            with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature importance
            try:
                with open(os.path.join(models_dir, 'feature_importance.pkl'), 'rb') as f:
                    importance_data = pickle.load(f)
                    self.feature_importance_score = importance_data['score']
                    self.feature_importance_pass_fail = importance_data['pass_fail']
                    self.feature_names = importance_data['names']
            except:
                pass
                
            print("✅ All models loaded successfully!")
        except FileNotFoundError:
            print("⚠️ Models not found. Training new models...")
            self.train_model()

# Test the models
if __name__ == "__main__":
    model = StudentPerformanceModel()
    model.train_model()
    
    # Test predictions
    print("\n" + "=" * 60)
    print("🧪 TEST PREDICTIONS")
    print("=" * 60)
    
    test_cases = [
        (95, 7, 85, "Excellent Student"),
        (75, 4, 60, "Average Student"),
        (50, 2, 35, "Struggling Student"),
    ]
    
    for attendance, study_hours, previous_marks, label in test_cases:
        print(f"\n📝 {label}:")
        print(f"   Input: Attendance={attendance}%, Study Hours={study_hours}h, Previous={previous_marks}%")
        
        # Random Forest prediction
        rf_score, rf_result, _ = model.predict(attendance, study_hours, previous_marks, use_random_forest=True)
        print(f"   🌲 Random Forest: Score={rf_score}%, Result={rf_result}")
        
        # Linear model prediction
        lr_score, lr_result, _ = model.predict(attendance, study_hours, previous_marks, use_random_forest=False)
        print(f"   📈 Linear Model: Score={lr_score}%, Result={lr_result}")
