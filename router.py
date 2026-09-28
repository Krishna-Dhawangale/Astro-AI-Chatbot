import joblib
import numpy as np

class IntentRouter:
    def __init__(self, model_path: str = "astrology_predict_model.pkl"):
        """
        Loads the pre-trained Logistic Regression model from the local environment.
        """
        try:
            self.model = joblib.load(model_path)
            print(f" Loaded model successfully from '{model_path}'")
        except Exception as e:
            print(f" Error loading model file '{model_path}': {e}")
            self.model = None

    def classify_and_predict(self, query: str, astro_features: list) -> dict:
        """
        Runs prediction using the trained model on incoming chart features.
        """
        if self.model is None:
            return {"predicted_domain": "general", "confidence_score": 0.0}

        # Convert features to 2D numpy array for scikit-learn
        feature_vector = np.array(astro_features).reshape(1, -1)
        
        # Predict class and probability
        prediction = self.model.predict(feature_vector)[0]
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0

        return {
            "predicted_domain": str(prediction),
            "confidence_score": confidence
        }