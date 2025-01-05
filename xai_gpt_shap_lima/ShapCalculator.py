import shap
import pandas as pd
import pickle
import os

class ShapCalculator:
    def __init__(self, model_path=None, data_path=None, target_class=None):
        """
        ShapCalculator class initialization
        
        Parameters:
        - model_path: Path to model (str)
        - data_path: Path to data(str)
        - target_class: Target class for which we want to analyze SHAP results (int)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.target_class = target_class
        self.model = None
        self.data = None
        self.shap_results = None 

    def load_model(self, model_path=None):
        """
        Load model from path.
        """
        if model_path:
            self.model_path = model_path
        if not self.model_path:
            raise ValueError("Path to model is not set.")
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def load_data(self, data_path=None):
        """
        Load data from path
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            raise ValueError("Path to data is not given.")
        self.data = pd.read_csv(self.data_path)

    def set_target_class(self, target_class):
        """
        Set target class for SHAP analysis
        """
        self.target_class = target_class

    def calculate_shap_values_for_instance(self, instance):
        """
        Calculate SHAP values and return them 
        """
        if self.model is None:
            raise ValueError("Model is not loaded")
        if self.data is None:
            raise ValueError("Data is not loaded.")
        if self.target_class is None:
            raise ValueError("Target class is not set.")

        explainer = shap.Explainer(self.model.predict_proba, self.data)
        shap_values = explainer(instance)
        shap_values_for_class = shap_values[..., self.target_class]
        
        self.shap_results = pd.DataFrame({
            "Feature": self.data.columns,
            "SHAP Value": shap_values_for_class.values[0],
            "Feature Value": instance.values[0]
        })
        return self.shap_results

    def save_shap_values_to_csv(self, output_path):
        """
        Save shap values to csv.
        """
        if self.shap_results is None:
            raise ValueError("SHAP are not available. Try running the SHAP analysis first")
        self.shap_results.to_csv(output_path, index=False)
        print(f"SHAP results were save to {output_path}")

    def calculate_shap_values_for_instance_old(self, instance):
        explainer = shap.Explainer(self.model.predict_proba,self.data)
        shap_values = explainer(instance)
        shap_values = shap_values[..., self.target_class]
        return shap_values
