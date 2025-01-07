import shap
import pandas as pd
import pickle
import onnxruntime

class ShapCalculator:

    """
    A class for calculating SHAP (SHapley Additive exPlanations) values
    for a machine learning model.

    Attributes:
        model_path (str): Path to the saved model.
        data_path (str): Path to the dataset.
        target_class (int): Target class for SHAP analysis (used for multi-class problems).
        model (object): Loaded machine learning model.
        data (DataFrame): Loaded dataset.
        shap_results (DataFrame): DataFrame containing SHAP values and feature contributions.
        model_type (str): Type of the loaded moddel (onnx, pickle or unknown)
    """
    def __init__(self, model_path=None, data_path=None, target_class=None):
        """
        Initializes the ShapCalculator class.

        Args:
            model_path (str, optional): Path to the saved model.
            data_path (str, optional): Path to the dataset (CSV).
            target_class (int, optional): Target class for SHAP analysis.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.target_class = target_class
        self.model = None
        self.data = None
        self.shap_results = None 
        self.model_type = None

    def load_model(self, model_path=None):
        """
        Loads a machine learning model from the specified file.

        Args:
            model_path (str, optional): Path to the model file. If not provided, uses `self.model_path`.

        Raises:
            ValueError: If the `model_path` is not set or the file cannot be loaded.
        """
        if model_path:
            self.model_path = model_path
        if not self.model_path:
            raise ValueError("Path to model is not set.")
        try:
            # Check if the model is an ONNX model
            if self.model_path.endswith(".onnx"):
                self.model = onnxruntime.InferenceSession(self.model_path)
                self.model_type = "onnx"
            else:
                # Asume that the model is a pickle file
                with open(self.model_path, "rb") as file:
                    self.model = pickle.load(file)
                    self.model_type = "pickle"
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
        
    def load_data(self, data_path=None):
        """
        Loads a dataset from a CSV file.

        Args:
            data_path (str, optional): Path to the dataset file. If not provided, uses `self.data_path`.

        Raises:
            ValueError: If the `data_path` is not set or the file cannot be loaded.
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            raise ValueError("Path to data is not given.")
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {e}")

    def set_target_class(self, target_class):
        """
        Sets the target class for SHAP analysis.

        Args:
            target_class (int): The target class index for SHAP analysis.
        """
        self.target_class = target_class

    def calculate_shap_values_for_instance(self, instance):
        """
        Calculates SHAP values for a single instance.

        Args:
            instance (DataFrame): A single instance (row) in the form of a pandas DataFrame. 
                                Ensure the instance matches the feature structure of the model's training data.

        Returns:
            Tuple:
                - DataFrame: A DataFrame containing SHAP values, feature names, and instance feature values.
                - numpy.ndarray: A 1D array containing SHAP values for the specified target class. 
                                This is directly compatible with SHAP visualization methods such as `shap.plots.waterfall`.

        Raises:
            ValueError: If any of the following conditions are not met:
                        - The model is loaded.
                        - The data (training set) is loaded.
                        - The target class is specified.
                        - The model type is supported (e.g., "onnx" or "pickle").
        """
        if self.model is None:
            raise ValueError("Model is not loaded")
        if self.data is None:
            raise ValueError("Data is not loaded.")
        if self.target_class is None:
            raise ValueError("Target class is not set.")
        
        if self.model_type == "onnx":
            input_name = self.model.get_inputs()[0].name
            pred_func = lambda x: self.model.run(None,{input_name: x.astype('float32')})[0]
        elif self.model_type == "pickle":
            if not hasattr(self.model, "predict_proba"):
                raise ValueError("Pickle model does not support 'predict_proba'.")
            pred_func = self.model.predict_proba
        else:
            raise ValueError("Unsupported model type: {self.model_type}")

        explainer = shap.Explainer(pred_func, self.data)
        shap_values = explainer(instance)
        shap_values_for_class = shap_values[..., self.target_class]
        
        self.shap_results = pd.DataFrame({
            "Feature": self.data.columns,
            "SHAP Value": shap_values_for_class.values[0],
            "Feature Value": instance.values[0]
        })

        return self.shap_results, shap_values_for_class 

    def save_shap_values_to_csv(self, output_path):
        """
        Saves the SHAP results to a CSV file.

        Args:
            output_path (str): Path to save the results.

        Raises:
            ValueError: If SHAP results are not calculated.
        """
        if self.shap_results is None:
            raise ValueError("SHAP are not available. Try running the SHAP analysis first")
        self.shap_results.to_csv(output_path, index=False)
        print(f"SHAP results were save to {output_path}")
