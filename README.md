# **xai-gpt-shap**

**xai-gpt-shap** is a Python library that combines **SHAP** (SHapley Additive exPlanations) value analysis with **OpenAI GPT**-based explanations to make machine learning model predictions more interpretable.

This library allows you to:
- Perform SHAP analysis on machine learning models.
- Generate role-specific explanations for SHAP results using OpenAI GPT (e.g., for beginners, analysts, or researchers).
- Interactively explore and understand SHAP results via a command-line interface (CLI).

---

## **Key Features**

- **SHAP Integration**: Calculate SHAP values for any machine learning model and dataset.
- **OpenAI GPT Integration**: Automatically explain SHAP results using OpenAI GPT with role-specific messages (e.g., beginner, analyst, executive summary).
- **Interactive Chat**: Engage in an interactive conversation with GPT to further explore results.
- **CLI Support**: Easily run SHAP analysis and explanations directly from the command line.

---

## **Installation**

### Option 1: Clone Repository (with Poetry)
This project is currently available as a Poetry project and not a published package. To use it:

1. Clone the repository:
   ```bash
   git clone https://github.com/JanHuntersi/xai-gpt-shap-lima.git
   cd xai-gpt-shap-lima
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   To include development dependencies (e.g., **XGBoost** is needed if you load model from data folder):
   ```bash
   poetry install --with dev
   ```

3. Run the CLI or use the library programmatically as described below.

---

### Option 2: Use Requirements File (pip)
If you prefer, you can use a `requirements.txt` file to install the dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/JanHuntersi/xai-gpt-shap-lima.git
   cd xai-gpt-shap-lima
   ```

2. Install dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the CLI or use the library programmatically as described below.

---

## **Usage**

### **1. Command-Line Interface (CLI)**

After installing the package, you can run it directly from the terminal using the `xai-gpt-shap` command.

#### **Example:**
```bash
xai-gpt-shap --api_key YOUR_API_KEY \
         --model_path model.pkl \
         --data_path data.csv \
         --instance_path instance.csv \
         --target_class 1 \
         --role beginner
```

#### **Options:**
- `--api_key`: Your OpenAI API key.
- `--model_path`: Path to the saved machine learning model (e.g., `model.pkl` or `model.onnx`).
- `--data_path`: Path to the dataset used for SHAP analysis (e.g., `data.csv`).
- `--instance_path`: Path to a CSV file containing the instance to analyze (e.g., `instance.csv`).
- `--target_class`: The target class for SHAP analysis (e.g., `1` for binary classification).
- `--role`: Role for the GPT explanation (`beginner`, `student`, `analyst`, `researcher`, `executive_summary`).
- `--interactive`: Enable interactive chat mode after the initial explanation.

#### **Example with Test Data:**
This library includes a `data` folder with prepared test datasets. To run an example:
1. Install the library with additional dependencies (for **XGBoost** support):
   ```bash
   poetry install --with dev
   ```
2. Move inside to xai_gpt_shap folder
   ```bash
   cd xai_gpt_shap
   ```


2. Run the project with the provided test data:
   ```bash
   poetry run python main.py \
       --model_path ../data/input/shap_model.pkl \
       --data_path ../data/input/x_data.csv \
       --instance_path ../data/input/selected_instance.csv \
       --target_class 1 \
       --output_csv ../data/output/output_csv.csv \
       --role beginner \
       --api_key YOUR_API_KEY
   ```

---

### **2. Programmatic Usage**

You can also use the library programmatically in Python scripts.

#### **Example Code:**
```python
from xai_gpt_shap import ChatGptClient, ShapCalculator

# Initialize the SHAP calculator
calculator = ShapCalculator(model_path="model.pkl", data_path="data.csv", target_class=1)
calculator.load_model()
calculator.load_data()

# Select an instance for SHAP analysis
selected_instance = calculator.data.iloc[[0]]  # First instance
shap_results = calculator.calculate_shap_values_for_instance(selected_instance)

# Initialize ChatGPT client
gpt_client = ChatGptClient(api_key="YOUR_API_KEY")

# Generate a role-specific explanation
message = gpt_client.create_summary_and_message(
    shap_df=shap_results,
    model="XGBoost",
    prediction_summary="Predicted income > 50k",
    target_class=1,
    role="beginner",
)
response = gpt_client.send_initial_prompt(message)

# Print the explanation
print(response)

# Start an interactive chat for follow-up questions
gpt_client.interactive_chat()
```

---

## **File Structure**

```
xai_gpt_shap/
│
├── xai_gpt_shap/        # Main package directory
│   ├── __init__.py          # Package initialization
│   ├── ChatGptClient.py     # Handles GPT interactions
│   ├── ShapCalculator.py    # Performs SHAP analysis
│   ├── roles.py             # Defines role-specific messages
│
├── tests/                   # Unit tests
│   ├── test_chat_gpt.py
│   ├── test_shap_calculator.py
│
├── README.md                # Documentation
├── pyproject.toml           # Poetry configuration
├── requirements.txt         # Alternative dependency file
└── LICENSE                  # License information
```

---

## **Supported Model Formats**

This library supports the following model formats:
1. **ONNX** (recommended): Platform-independent and standardized format.
2. **Pickle**: Python models saved with `pickle` (e.g., Scikit-learn, XGBoost).

**Note:** When using Pickle models, the user must ensure that the required libraries (e.g., `scikit-learn`, `xgboost`) are installed.

---

## **Key Methods**

Here’s a breakdown of key methods in the library:

| **Class**         | **Method**                              | **Description**                                                                                     | **Parameters**                                                                               | **Returns**                                                                                 |
|-------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `ChatGptClient`   | `send_initial_prompt(prompt)`           | Sends an initial prompt to OpenAI GPT and returns the assistant’s response.                         | `prompt` (str): The prompt to send to GPT.                                                 | `str`: The assistant’s response.                                                          |
|                   | `interactive_chat()`                   | Starts an interactive session with GPT for follow-up questions.                                    | None                                                                                       | None                                                                                       |
|                   | `create_summary_and_message(...)`      | Generates a GPT prompt from SHAP results, model details, and role-specific requirements.            | `shap_df` (DataFrame): SHAP values, `model` (str): Model name, `short_summary` (str): Summary, `choice_class` (str): Class name, `role` (str): Role. | `str`: Generated GPT prompt.                                                              |
|                   | `set_system_message(message)`          | Sets the system-level message to configure GPT’s behavior.                                          | `message` (str): System-level message.                                                    | None                                                                                       |
|                   | `stream_response()`                    | Streams GPT’s response in real time to the console.                                                | None                                                                                       | `str`: The full streamed response.                                                        |
|                   | `clean_chat_history(max_history_tokens)`| Cleans the chat history to reduce token count in case of large conversation contexts.               | `max_history_tokens` (int): Maximum tokens allowed in history.                            | None                                                                                       |
| `ShapCalculator`  | `load_model(model_path)`               | Loads a machine learning model from a file.                                                        | `model_path` (str): Path to the model file.                                               | None                                                                                       |
|                   | `load_data(data_path)`                 | Loads a dataset from a CSV file.                                                                   | `data_path` (str): Path to the dataset file.                                              | None                                                                                       |
|                   | `set_target_class(target_class)`       | Sets the target class for SHAP analysis (for multi-class problems).                                | `target_class` (int): The class index to analyze.                                         | None                                                                                       |
|                   | `calculate_shap_values_for_instance(instance)` | Calculates SHAP values for a given instance and returns a DataFrame with feature-level explanations. | `instance` (DataFrame): The instance for SHAP analysis.                                   | `DataFrame`: SHAP values with feature importance.                                         |
|                   | `save_shap_values_to_csv(output_path)` | Saves the SHAP values to a CSV file.                                                               | `output_path` (str): Path to save the CSV file.                                           | None                                                                                       |
|                   | `get_feature_importance()`             | Computes and summarizes feature importance across all instances in the dataset.                     | None                                                                                       | `DataFrame`: Feature importance summary.                                                  |
