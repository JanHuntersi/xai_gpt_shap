import shap
import xgboost
import pandas as pd
import os

#TODO model agnostic


class ShapCalculator:
    def __init__(self):
        self.data_path= os.path.join(os.path.dirname(__file__), "data", "10_instance.csv")
        self.model = None
        self.x_data = None
        self.shap_values = None
        self.chosen_instance = None
        self.shap = shap


    def test_get_shap_values(self): 
        shap_positives = pd.read_csv(self.data_path)
        return shap_positives
    

    def calculate_shap_values(self):
        X, y = shap.datasets.adult()


        print("ucenje modela")
        model = xgboost.XGBClassifier()
        model.fit(X, y);

        explainer = shap.explainers.Exact(model.predict_proba, X)
        shap_values = explainer(X[:100])

        positive_shap_values = shap_values[..., 1]   #Pozitivni razred
        chosen_instance = positive_shap_values[0]

        self.model = model
        self.x_data = X
        self.shap_values = shap_values
        self.chosen_instance = chosen_instance
        self.shap = shap
        
        return self.shap_values, self.chosen_instance, self.shap
