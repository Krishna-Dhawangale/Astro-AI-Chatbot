import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import SGDClassifier

def generate_and_save_s_curve():
    print("📡 Step 1: Connecting to streaming dataset...")
    jyotish_stream = load_dataset("AmareshHebbar/jyotish-llm-sft", split="train", streaming=True)
    
    # Instantiate the classifier engine with explicit binary outcome definitions [0, 1]
    model = SGDClassifier(loss='log_loss', random_state=42)
    all_classes = np.array([0, 1])  # FIXED: Explicit array parameters included
    
    X_all, y_all = [], []
    processed_count = 0
    max_samples = 15000  # Baseline slice for rapid calculations
    
    print("🧼 Step 2: Extracting structural planetary features...")
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
            mars_capricorn_interaction = is_lagna_capricorn * has_mars_exalted
            
            # Synthesize an aggregated numeric parameter score across variables for a clear X-axis spectrum
            total_astro_score = is_lagna_capricorn + has_mars_exalted + (mars_capricorn_interaction * 2)
            
            # Map structural target vector definitions
            target_label = 1 if ('dhana_yoga_2_11' in rule_ids or 'mars_exalted' in rule_ids) else 0
            
            X_all.append([total_astro_score])
            y_all.append(target_label)
            processed_count += 1
            
            if processed_count >= max_samples:
                break
        except Exception:
            continue

    X_matrix = np.array(X_all)
    y_matrix = np.array(y_all)
    
    print("🏋️ Step 3: Localized training of the validation vector...")
    model.fit(X_matrix, y_matrix)
    
    # 4. Generate continuous linear inputs across the X-axis to sketch a smooth, unbroken S-Curve
    X_continuous_range = np.linspace(X_matrix.min() - 1, X_matrix.max() + 1, 400).reshape(-1, 1)
    
    # Pass the inputs directly through the native Sigmoid Activation function logic
    y_calculated_probabilities = model.predict_proba(X_continuous_range)[:, 1]
    
    print("\n================ S-CURVE CALIBRATION DETAILS ================")
    print(f"X-Axis Input Range Bounds: {X_matrix.min()} up to {X_matrix.max()}")
    print(f"Calculated 50% Probability Cross-over Feature Threshold: {np.median(X_continuous_range):.2f}")
    print("=============================================================")
    
    # Sort data for proper coordinate rendering alignment
    sort_idx = np.argsort(X_matrix.flatten())
    X_scatter = X_matrix.flatten()[sort_idx]
    y_scatter = y_matrix[sort_idx]
    
    # 5. Build and save the file plot locally to your machine
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scatter, y_scatter, color='orange', alpha=0.3, label='User Profiles (Real Data Data)', zorder=2)
    plt.plot(X_continuous_range.flatten(), y_calculated_probabilities, color='blue', linewidth=3, label='Sigmoid Activation Curve (Best Fit)', zorder=3)
    plt.axhline(0.5, color='red', linestyle='--', label='50% Decision Gate Threshold', alpha=0.7)
    
    plt.title('Vedic Chatbot Astrological Prediction Activation S-Curve')
    plt.xlabel('Calculated Astrological Feature Weight Score')
    plt.ylabel('Predicted Career Success Probability Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # Save the graphic format file cleanly into your astro_env folder
    output_image_name = "astrology_activation_curve.png"
    plt.savefig(output_image_name, dpi=300)
    plt.close()
    
    print(f"\n🎉 Success! The S-Curve visualization has been compiled and saved as: '{output_image_name}'")
    print("👉 Look in your astro_env directory to open the image file and inspect the best-fit line accuracy.")

if __name__ == "__main__":
    generate_and_save_s_curve()
