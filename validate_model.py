import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_validation_pipeline():
    print("📡 Step 1: Initializing secure streaming connection for testing data...")
    jyotish_stream = load_dataset("AmareshHebbar/jyotish-llm-sft", split="train", streaming=True)
    
    # 1. Initialize our Logistic Regression model
    model = SGDClassifier(loss='log_loss', random_state=42)
    all_classes = np.array([0, 1])
    
    # Data collections for splitting
    X_all = []
    y_all = []
    processed_count = 0
    max_samples = 50000 # Using a clean 50k dataset slice for quick validation testing
    
    print("🧼 Step 2: Extraction and feature encoding phase...")
    for row in jyotish_stream:
        if row['question_type'] != 'career':
            continue
        try:
            facts = row['chart_facts'] if isinstance(row['chart_facts'], dict) else json.loads(row['chart_facts'])
            rules = row['retrieved_rules'] if isinstance(row['retrieved_rules'], list) else json.loads(row['retrieved_rules'])
            
            # Feature extraction matching your exact blueprint
            lagna = facts.get('ascendant', {}).get('rashi', '')
            is_lagna_capricorn = 1 if lagna == 'Capricorn' else 0
            is_lagna_aries = 1 if lagna == 'Aries' else 0
            is_lagna_taurus = 1 if lagna == 'Taurus' else 0
            
            rule_ids = [rule.get('id') for rule in rules if isinstance(rule, dict)]
            has_budhaditya = 1 if 'budhaditya' in rule_ids else 0
            has_mars_exalted = 1 if 'mars_exalted' in rule_ids else 0
            has_moon_exalted = 1 if 'moon_exalted' in rule_ids else 0
            mars_capricorn_interaction = is_lagna_capricorn * has_mars_exalted
            
            features = [
                is_lagna_capricorn, is_lagna_aries, is_lagna_taurus,
                has_budhaditya, has_mars_exalted, has_moon_exalted,
                mars_capricorn_interaction
            ]
            
            target_label = 1 if ('dhana_yoga_2_11' in rule_ids or 'mars_exalted' in rule_ids or 'gaja_kesari' in rule_ids) else 0
            
            X_all.append(features)
            y_all.append(target_label)
            processed_count += 1
            
            if processed_count >= max_samples:
                break
        except Exception:
            continue

    # Convert everything to solid numerical matrices
    X_matrix = np.array(X_all)
    y_matrix = np.array(y_all)
    
    print("✂️ Step 3: Splitting data into Training (80%) and Testing (20%) matrices...")
    # Calculate the split index mathematically to keep memory footprint light
    split_index = int(len(X_matrix) * 0.8)
    
    X_train, X_test = X_matrix[:split_index], X_matrix[split_index:]
    y_train, y_test = y_matrix[:split_index], y_matrix[split_index:]
    
    print("🏋️ Step 4: Training model weights matrix on training slice...")
    model.fit(X_train, y_train)
    
    print("🧪 Step 5: Executing prediction validations on unseen hidden data...")
    # The model makes blind guesses on the remaining 20% test slice
    y_predictions = model.predict(X_test)
    
    # Calculate performance scores
    final_accuracy = accuracy_score(y_test, y_predictions)
    
    print("\n================ VALIDATION REPORT ================")
    print(f"🎯 Model Prediction Accuracy: {final_accuracy * 100:.2f}%")
    print("---------------------------------------------------")
    print(classification_report(y_test, y_predictions, target_names=['Friction Block', 'Success']))
    print("===================================================")

if __name__ == "__main__":
    run_validation_pipeline()
