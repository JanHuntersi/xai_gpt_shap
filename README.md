 # xai-gpt-shap-lima

**xai-gpt-shap-lima** is a Python library that combines SHAP (SHapley Additive exPlanations) value analysis with OpenAI GPT-based explanations to make machine learning model predictions more interpretable.

This library allows you to:
- Perform SHAP analysis on machine learning models.
- Generate role-specific explanations for SHAP results using OpenAI GPT (e.g., for beginners, analysts, or researchers).
- Interactively explore and understand SHAP results via a command-line interface (CLI).

## Features

- **SHAP Integration**: Calculate SHAP values for any machine learning model and dataset.
- **OpenAI GPT Integration**: Automatically explain SHAP results using OpenAI GPT with role-specific messages (beginner, analyst, executive, etc.).
- **Interactive Chat**: Engage in an interactive conversation with GPT to further explore results.
- **CLI Support**: Easily run SHAP analysis and explanations directly from the command line.

---

## Installation

### Using Poetry
If you use [Poetry](https://python-poetry.org/):
```bash
poetry add xai-gpt-shap-lima
```

### Using pip
Install the package using pip:
```bash
pip install xai-gpt-shap-lima
```

---

## Usage

### 1. **Command-Line Interface (CLI)**

After installing the package, you can run it directly from the terminal using the `xai-shap` command.

#### Example:
```bash
xai-shap --api_key YOUR_API_KEY \
         --model_path model.pkl \
         --data_path data.csv \
         --instance_path instance.csv \
         --target_class 1 \
         --role beginner
```

#### Options:
- `--api_key`: Your OpenAI API key.
- `--model_path`: Path to the saved machine learning model (e.g., `model.pkl`).
- `--data_path`: Path to the dataset used for SHAP analysis (e.g., `data.csv`).
- `--instance_path`: Path to a CSV file containing the instance to analyze (e.g., `instance.csv`).
- `--target_class`: The target class for SHAP analysis (e.g., `1` for binary classification).
- `--role`: Role for the GPT explanation (`beginner`, `student`, `analyst`, `researcher`, `executive_summary`).
- `--interactive`: Enable interactive chat mode after the initial explanation.

---

### 2. **Programmatic Usage**

You can also use the library programmatically in Python scripts.

#### Example Code:
```python
from xai_gpt_shap_lima import ChatGptClient, ShapCalculator

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

## File Structure

```
xai_gpt_shap_lima/
│
├── xai_gpt_shap_lima/        # Main package directory
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
└── LICENSE                  # License information
```

---

## Available Methods

Here’s a detailed breakdown of key methods in the library:

| **Class**         | **Method**                              | **Description**                                                                                     | **Parameters**                                                                               | **Returns**                                                                                 |
|-------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `ChatGptClient`   | `send_initial_prompt(prompt)`           | Sends an initial prompt to OpenAI GPT and returns the assistant’s response.                         | `prompt` (str): The prompt to send to GPT.                                                 | `str`: The assistant’s response.                                                          |
|                   | `interactive_chat()`                   | Starts an interactive session with GPT for follow-up questions.                                    | None                                                                                       | None                                                                                       |
|                   | `create_summary_and_message(...)`      | Generates a GPT prompt from SHAP results, model details, and role-specific requirements.            | `shap_df` (DataFrame): SHAP values, `model` (str): Model name, `short_summary` (str): Summary, `choice_class` (str): Class name, `role` (str): Role. | `str`: Generated GPT prompt.                                                              |
|                   | `set_gpt_expertise_layer(role)`        | Configures GPT’s system message and settings based on the selected role.                           | `role` (str): The role to set (e.g., beginner, analyst).                                   | `str`: The configured role.                                                               |
|                   | `set_system_message(message)`          | Sets the system-level message to configure GPT’s behavior.                                          | `message` (str): System-level message.                                                    | None                                                                                       |
|                   | `stream_response()`                    | Streams GPT’s response in real time to the console.                                                | None                                                                                       | `str`: The full streamed response.                                                        |
|                   | `clean_chat_history(history_tokens=0)` | Cleans chat history to keep the token count within limits for smoother interactions.               | `history_tokens` (int, optional): Maximum tokens allowed in history.                      | None                                                                                       |
| `ShapCalculator`  | `load_model(model_path)`               | Loads a machine learning model from a file.                                                        | `model_path` (str): Path to the model file.                                               | None                                                                                       |
|                   | `load_data(data_path)`                 | Loads a dataset from a CSV file.                                                                   | `data_path` (str): Path to the dataset file.                                              | None                                                                                       |
|                   | `set_target_class(target_class)`       | Sets the target class for SHAP analysis (for multi-class problems).                                | `target_class` (int): The class index to analyze.                                         | None                                                                                       |
|                   | `calculate_shap_values_for_instance(instance)` | Calculates SHAP values for a given instance and returns a DataFrame with feature-level explanations. | `instance` (DataFrame): The instance for SHAP analysis.                                   | `DataFrame`: SHAP values with feature importance.                                         |
|                   | `save_shap_values_to_csv(output_path)` | Saves the SHAP values to a CSV file.                                                               | `output_path` (str): Path to save the CSV file.                                           | None                                                                                       |
| `roles`           | `get_role_message(role)`               | Retrieves a predefined system message based on the user’s expertise level or role.                 | `role` (str): The role to retrieve the message for (e.g., beginner, analyst).             | `str`: Predefined system message.                                                         |



---

## Dependencies

The library requires the following dependencies, which will be installed automatically:
- `openai` (OpenAI API integration)
- `shap` (SHAP value computation)
- `numpy`, `pandas`, `scikit-learn` (Data manipulation and ML model support)
- `rich`, `prompt-toolkit` (Interactive terminal output)
- `python-dotenv` (Environment variable management)
- `tiktoken` (Token handling for OpenAI API)

---

## Development

If you'd like to contribute or modify the library, clone the repository and install the dependencies using Poetry:
```bash
git clone https://github.com/your-repo/xai-gpt-shap-lima.git
cd xai-gpt-shap-lima
poetry install
```

Run tests to ensure everything works:
```bash
poetry run pytest
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Credits

Developed by **Jan Sernec (JanHuntersi)**.
