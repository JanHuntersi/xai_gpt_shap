from openai import OpenAI
import os
from dotenv import load_dotenv

from xai_gpt_shap_lima.ChatGptClient import ChatGptClient
from xai_gpt_shap_lima.ShapCalculator import ShapCalculator

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


api_key = os.getenv("OPENAI_API_KEY")
gpt_client = ChatGptClient(api_key)

shap_calculator = ShapCalculator()



def test_chat_gpt_client():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Tell me its gonna be alright",
            }
        ]
    )

    print(completion.choices[0].message)


def test_chat_gpt_client():
    shap_df = shap_calculator.test_get_shap_values()
    print("SHAP vrednosti:")

    # Ustvari povzetek
    summary = "\n".join(
        [
            f"- {row['Feature']}: SHAP={row['SHAP Value']:.4f}, Value={row['Feature Value']}"
            for _, row in shap_df.iterrows()
        ]
    )
    # Prepare the prompt for ChatGPT
    message = f"""
    Imam razlago, ki temelji na SHAP vrednostih za eno instanco. Model je XGBoost in napoveduje, ali oseba zasluži več kot 50k na leto. Tukaj so rezultati za izbrano instanco pozitivnega razreda:

    {summary}

    Prosimo, da mi podate razlago, kako te značilke vplivajo na napoved.
    """

    print(message)

    # Send the message to ChatGPT API
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are an assistant specialized in interpreting SHAP analysis for a single instance. Help the user understand why the model made a specific decision by analyzing the SHAP values. Provide explanations in a way that assumes familiarity with the domain but not with SHAP itself. Assume the selected outcome is 'True'."},
            {"role": "user", "content": message},
        ],
    )
    
        # Print the response
    print("ChatGPT Explanation:")
    print(response.choices[0].message.content)


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


def main():

    gpt_client.custom_console_message("Calculating SHAP values...")
    shap_df = shap_calculator.test_get_shap_values()
    gpt_client.custom_console_message("SHAP values calculated. Sendimg them results to model..")

    
    gpt_client.set_system_message("Ti si sistem ki pomagala razbrati SHAP vrednosti, in na podlagi teh shap vrednosti pomagas uporabniku razumeti, zakaj je točno ta odločitev bila narejena za eno instanco. Odgovarjaj tak kot če bi odgovarjal nekomu ki se na tehniko spozna, ne pa na SHAP vrednosti")



    message = create_summary_and_message(shap_df, "XGBoost", "ali oseba zasluži več kot 50k na leto", "pozitivnega")
    

    answer = gpt_client.send_initial_prompt(message)

    gpt_client.interactive_chat()

    print("konec")


if __name__ == "__main__":
    main()




