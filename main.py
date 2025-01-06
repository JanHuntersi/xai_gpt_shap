"""
TODO's:
- Odstrani pirates.
- ...
Extra TODO's:
- Send initial prompt da vrne bare text, mogoce knjiznjica za delo z lepimi izpisi?
- Uporabnik ročno vnese vrednosti instance
- Check if API key is correct
- Poti ki jih nastavi uporabnik naj bodo neodvisne
"""
import argparse
import pandas as pd
from xai_gpt_shap_lima.ChatGptClient import ChatGptClient
from xai_gpt_shap_lima.ShapCalculator import ShapCalculator



def parse_arguments():    
    """
    Reads arguments from the command line and returns them.
    """
    parser = argparse.ArgumentParser(description="Izvede SHAP analizo za podano instanco.")
    parser.add_argument("--model_path", required=True, help="Pot do shranjenega modela (npr. shap_model.pkl)")
    parser.add_argument("--data_path", required=True, help="Pot do podatkov (npr. shap_dataset.csv)")
    parser.add_argument("--instance_path", required=True, help="Pot do datoteke z izbrano instanco (npr. selected_instance.csv)")
    parser.add_argument("--target_class", type=int, required=False, help="Ciljni razred za SHAP analizo (npr. 1)")
    parser.add_argument("--output_csv", required=False, help="Pot za shranjevanje SHAP rezultatov (npr. shap_results.csv)")
    parser.add_argument("--role", required=False, help="Izberi vlogo: beginner, student, analyst, researcher, executive_summary")
    parser.add_argument("--api_key", required=True, help="Api key for OpeanAi")
    return parser.parse_args()

def main():

    args = parse_arguments()

    gpt_client = ChatGptClient(args.api_key)
    calculator = ShapCalculator()

    calculator.load_model(args.model_path)
    calculator.load_data(args.data_path)
    calculator.set_target_class(args.target_class)

    # load selected instance on which SHAP analysis should be run
    selected_instance = pd.read_csv(args.instance_path)
    shap_results = calculator.calculate_shap_values_for_instance(selected_instance)

    gpt_client.custom_console_message("Calculating SHAP values..." )
    gpt_client.custom_console_message("SHAP values calculated. Sending them gpt..")
    
    #try setting roles
    try:
        role = gpt_client.select_gpt_role(args.role if hasattr(args, "role") else None)
        gpt_client.custom_console_message(f"Using role: {role.capitalize()}", "green")
    except ValueError as e:
        gpt_client.custom_console_message(f"[red]Failed to set GPT expertise layer: {e}[/red]")
        exit(1)

    message = gpt_client.create_summary_and_message(shap_results, "XGBoost", "ali oseba zasluži več kot 50k na leto", "pozitivnega", role)
 
    gpt_client.send_initial_prompt(message, max_response_tokens = 500)
    gpt_client.interactive_chat()


if __name__ == "__main__":
    main()
