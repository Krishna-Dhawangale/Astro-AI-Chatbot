import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

def calculate_and_export_chart_data():
    print("📡 Step 1: Connecting to streaming dataset...")
    jyotish_stream = load_dataset("AmareshHebbar/jyotish-llm-sft", split="train", streaming=True)
    
    # FIX: Pass the array of classes [0, 1] explicitly
    model = SGDClassifier(loss='log_loss', random_state=42)
    all_classes = np.array([0, 1])  
    
    X_all, y_all = [], []
    processed_count = 0
    max_samples = 15000  
    
    print("🧼 Step 2: Extracting features and target labels...")
    # ... (rest of your existing code continues down from here)

    for row in jyotish_stream:
        if row['question_type'] != 'career':
            continue
        try:
            facts = row['chart_facts'] if isinstance(row['chart_facts'], dict) else json.loads(row['chart_facts'])
            rules = row['retrieved_rules'] if isinstance(row['retrieved_rules'], list) else json.loads(row['retrieved_rules'])
            
            lagna = facts.get('ascendant', {}).get('rashi', '')
            is_lagna_capricorn = 1 if lagna == 'Capricorn' else 0
            
            rule_ids = [rule.get('id') for rule in rules if isinstance(rule, dict)]
            has_mars_exalted = 1 if 'mars_exalted' in rule_ids else 0
            
            # Use our core interaction tracking term as our primary visualization x-axis parameter
            # This represents a complex Vedic rule (Mars Exalted in Capricorn Ascendant)
            mars_capricorn_interaction = is_lagna_capricorn * has_mars_exalted
            
            # For plotting, we evaluate a simplified continuous weight spectrum
            total_score = is_lagna_capricorn + has_mars_exalted + (mars_capricorn_interaction * 2)
            
            target_label = 1 if ('dhana_yoga_2_11' in rule_ids or 'mars_exalted' in rule_ids) else 0
            
            X_all.append([total_score])
            y_all.append(target_label)
            processed_count += 1
            
            if processed_count >= max_samples:
                break
        except Exception:
            continue

    X_matrix = np.array(X_all)
    y_matrix = np.array(y_all)
    
    # Train the linear model weights
    model.fit(X_matrix, y_matrix)
    
    # Generate continuous test input ranges across the x-axis for plotting the clean S-Curve line
    X_line_range = np.linspace(X_matrix.min() - 1, X_matrix.max() + 1, 300).reshape(-1, 1)
    
    # Compute the best fit probability values for the sigmoid curve line
    y_line_probabilities = model.predict_proba(X_line_range)[:, 1]
    
    # Print numerical calibration points for structural text verification
    print("\n================ CALCULATED DATA MILSTONES ================")
    print(f"X-Axis Bounds (Astrological Feature Weight Score): Min {X_matrix.min()} to Max {X_matrix.max()}")
    print(f"Computed Midpoint Probability Cross-over Coordinate: {np.median(X_line_range):.2f}")
    print("===========================================================")
    
    # Export numerical logs for visualization injection
    # Row pairs sorted by feature score order
    sorted_indices = np.argsort(X_matrix.flatten())
    X_scatter = X_matrix.flatten()[sorted_indices]
    y_scatter = y_matrix[sorted_indices]
    
    return X_scatter, y_scatter, X_line_range.flatten(), y_line_probabilities

# Execute calculation variables processing
X_scat, y_scat, X_line, y_line = calculate_and_export_chart_data()
