from openai import OpenAI
import os
from dotenv import load_dotenv

from xai_gpt_shap_lima.ChatGptClient import ChatGptClient
from xai_gpt_shap_lima.ShapCalculator import ShapCalculator
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gpt_client = ChatGptClient(api_key)
shap_calculator = ShapCalculator()




def create_summary_and_message(shap_df,model,short_summary, choice_class):
    summary = "\n".join(
        [
            f"- {row['Feature']}: SHAP={row['SHAP Value']:.4f}, Value={row['Feature Value']}"
            for _, row in shap_df.iterrows()
        ]
    )
    # Prepare the prompt for ChatGPT
    message = f"""
    Imam razlago, ki temelji na SHAP vrednostih za eno instanco. Model je {model} in napoveduje: {short_summary}. Tukaj so rezultati za izbrano instanco {choice_class} razreda:

    {summary}

    Prosimo, da mi podate razlago, kako te značilke vplivajo na napoved.
    """



    return message

#TODO izboljsaj izpis in izgled

def choose_gpt_expertise_layer():

    BASE_MESSAGE = "You are a system designed to interpret SHAP values and help the user understand why a particular decision was made for a specific instance based on these SHAP values."

    while True:
        gpt_client.custom_console_message("Please choose the ChatGPT expertise level you want to interact with:")
        gpt_client.custom_console_message("1. Explain it to me like I'm five") 
        gpt_client.custom_console_message("2. Explain it to me like I'm a student")
        gpt_client.custom_console_message("3. Explain it to me like I'm a professor")
        gpt_client.custom_console_message("4. Exit")
        
        choice = gpt_client.get_user_input()

        if choice == "1":
            gpt_client.custom_console_message("You choose the expertise level: Explain it to me like I'm five")
            gpt_client.set_system_message(BASE_MESSAGE + "You will explain SHAP values to a five-year-old.")
            gpt_client.set_temperature(1)
            gpt_client.set_tokens(400)
            break
        elif choice == "2":
            gpt_client.custom_console_message("You choose the expertise level: Explain it to me like I'm a student")
            gpt_client.set_system_message(BASE_MESSAGE  + "You will explain SHAP values to a student.")
            gpt_client.set_temperature(0.5)
            gpt_client.set_tokens(350)
            break
        elif choice == "3":
            gpt_client.custom_console_message("You choose the expertise level: Explain it to me like I'm a professor")
            gpt_client.set_system_message(BASE_MESSAGE + "You will explain SHAP values to a professor.")
            gpt_client.set_temperature(0.1)
            gpt_client.set_tokens(200)
            break
        elif choice == "0":
            gpt_client.custom_console_message("You choose the SECRET expertise level: Explain it to me like I'm a pirate")
            gpt_client.set_system_message(BASE_MESSAGE + "Now you are captain Jack Sparrow, You will explain SHAP values to me like I'm Will Turner.")
            gpt_client.set_temperature(0.5)
            gpt_client.set_tokens(500)
            break
        elif choice == "4":
            gpt_client.custom_console_message("Exiting chat. Goodbye!")
            exit()


def main():

    gpt_client.custom_console_message("Calculating SHAP values..." )
    shap_df = shap_calculator.test_get_shap_values()
    gpt_client.custom_console_message("SHAP values calculated. Sendimg them results to model..")

    #Defualt value
    gpt_client.set_system_message("Ti si sistem ki pomagala razbrati SHAP vrednosti, in na podlagi teh shap vrednosti pomagas uporabniku razumeti, zakaj je točno ta odločitev bila narejena za eno instanco. Odgovarjaj tak kot če bi odgovarjal nekomu ki se na tehniko spozna, ne pa na SHAP vrednosti")
    
    #Choose the expertise level
    choose_gpt_expertise_layer() 


    message = create_summary_and_message(shap_df, "XGBoost", "ali oseba zasluži več kot 50k na leto", "pozitivnega")
    
    gpt_client.send_initial_prompt(message, max_tokens = 500)

    gpt_client.interactive_chat()

    gpt_client.custom_console_message("KONEC..")


if __name__ == "__main__":
    main()




