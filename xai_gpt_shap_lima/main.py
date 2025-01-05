"""
TODO's:
- Naredi v knjiznjico, izpili prompte in izpise.
- ...
Extra TODO's:
- Uporabnik ročno vnese vrednosti instance
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import argparse
import pandas as pd
from xai_gpt_shap_lima.ChatGptClient import ChatGptClient
from xai_gpt_shap_lima.ShapCalculator import ShapCalculator
from xai_gpt_shap_lima.roles import get_role_message
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gpt_client = ChatGptClient(api_key)
shap_calculator = ShapCalculator()


def create_summary_and_message(shap_df, model, short_summary, choice_class, role):

    summary = "\n".join(
        [
            f"- {row['Feature']}: SHAP={row['SHAP Value']:.4f}, Value={row['Feature Value']}"
            for _, row in shap_df.iterrows()
        ]
    )
    
    top_positive = shap_df.nlargest(3, "SHAP Value")
    top_negative = shap_df.nsmallest(3, "SHAP Value")
    
    top_positive_summary = "\n".join(
        [
            f"- {row['Feature']}: SHAP={row['SHAP Value']:.4f}, Value={row['Feature Value']}"
            for _, row in top_positive.iterrows()
        ]
    )
    
    top_negative_summary = "\n".join(
        [
            f"- {row['Feature']}: SHAP={row['SHAP Value']:.4f}, Value={row['Feature Value']}"
            for _, row in top_negative.iterrows()
        ]
    )

    # Prompt bassed on role
    if role == "beginner":
        message = f"""
        Imagine you are explaining SHAP values to a beginner. 
        The model predicts: {short_summary}. 
        Focus only on the most important features and their effects. Avoid using numbers.

        Key Insights:
        The most important positive feature is {top_positive.iloc[0]['Feature']}.
        The most important negative feature is {top_negative.iloc[0]['Feature']}.

         Full SHAP Results:
        {summary}


        Explain this prediction in simple terms.
        """
    elif role == "executive_summary":
        message = f"""
        Provide a concise summary of the SHAP values for the prediction: {short_summary}.
        Focus on the most important features and their contributions without technical details.

        Key Insights:
        Positive: {top_positive.iloc[0]['Feature']} (positive impact).
        Negative: {top_negative.iloc[0]['Feature']} (negative impact).

        Full SHAP Results:
        {summary}

        """
    else:  # Default for other roles
        message = f"""
        I have an explanation based on SHAP values for a single instance. 
        The model used is {model}, and it predicts: {short_summary}. 
        Below are the SHAP values for the chosen instance from the {choice_class} class:

        Full SHAP Results:
        {summary}

        Key Insights:
        The top 3 features positively influencing the prediction are:
        {top_positive_summary}

        The top 3 features negatively influencing the prediction are:
        {top_negative_summary}

        Please analyze the SHAP results and explain:
        1. How these features contribute to the prediction.
        2. Why the prediction was made for this specific instance.
        3. Any potential insights or counterintuitive results.

        Use clear and concise language based on the expertise level selected earlier.
        """
    
    return message

def choose_gpt_expertise_layer():

    roles_config = {
            "1": {"role": "beginner", "temperature": 1, "tokens": 300, "message": "Explain it to me like I'm a beginner"},
            "2": {"role": "student", "temperature": 0.8, "tokens": 200, "message": "Explain it to me like I'm a student"},
            "3": {"role": "analyst", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like I'm an analyst"},
            "4": {"role": "researcher", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like I'm a researcher"},
            "5": {"role": "executive_summary", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like an executive summary"},
            "0": {"role": "pirate", "temperature": 0.8, "tokens": 200, "message": "Explain it to me like I'm a pirate"},
        }

    while True:

        gpt_client.custom_console_message("Please choose the ChatGPT expertise level you want to interact with:")
        gpt_client.custom_console_message("1. Beginner (Explain it to me like I'm a beginner)")
        gpt_client.custom_console_message("2. Student (Explain it to me like I'm a student)")
        gpt_client.custom_console_message("3. Analyst (Explain it to me like I'm an analyst)")
        gpt_client.custom_console_message("4. Researcher (Explain it to me like I'm a researcher)")
        gpt_client.custom_console_message("5. Executive Summary (Explain it to me like an executive summary)")
        gpt_client.custom_console_message("6. Exit")
        
        
        choice = gpt_client.get_user_input()

        if choice in roles_config:
            
            config = roles_config[choice]
            role = config["role"]
            gpt_client.set_temperature(config["temperature"])
            gpt_client.set_tokens(config["tokens"])
            gpt_client.custom_console_message(f"You chose the expertise level: {config['message']}", "yellow")
            
            # Set system msg
            try:
                system_message = get_role_message(role)
                gpt_client.set_system_message(system_message)
                return role  
            except ValueError as e:
                gpt_client.custom_console_message(f"[red]{e}[/red]")
        elif choice == "6":
            gpt_client.custom_console_message("Exiting chat. Goodbye!", "red")
            exit()
        else:
            gpt_client.custom_console_message("Invalid choice. Please try again.", "red")

def set_gpt_expertise_layer(role=None):
    """
    Nastavi sistemsko sporočilo GPT na podlagi izbrane vloge.

    Args:
        role (str): Izbrana vloga (npr. "beginner", "student", "analyst").
                    Če ni podana, se interaktivno vpraša uporabnika.

    Returns:
        str: Izbrano vlogo (npr. "beginner", "student").
    """
    if role:
        # if role available we set it
        try:
            system_message = get_role_message(role)
            gpt_client.set_system_message(system_message)
            gpt_client.custom_console_message(f"Selected role: {role.capitalize()}", "green")
            return role
        except ValueError as e:
            gpt_client.custom_console_message(f"[red]{e}[/red]")
            raise e
    else:
        # if role not set use interacitve way
        return choose_gpt_expertise_layer()

        

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
    return parser.parse_args()

def main():



    args = parse_arguments()

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
        role = set_gpt_expertise_layer(args.role if hasattr(args, "role") else None)
        gpt_client.custom_console_message(f"Using role: {role.capitalize()}", "green")
    except ValueError as e:
        gpt_client.custom_console_message(f"[red]Failed to set GPT expertise layer: {e}[/red]")
        exit(1)

    message = create_summary_and_message(shap_results, "XGBoost", "ali oseba zasluži več kot 50k na leto", "pozitivnega", role)
   
    gpt_client.send_initial_prompt(message, max_tokens = 500)

    gpt_client.interactive_chat()

    gpt_client.custom_console_message("KONEC..")


if __name__ == "__main__":
    main()




# BASE_MESSAGE = "You are a system designed to interpret SHAP values and help the user understand why a particular decision was made for a specific instance based on these SHAP values. Do not provide the SHAP values themselves unless asked, but rather explain the impact of these features on the prediction. "
