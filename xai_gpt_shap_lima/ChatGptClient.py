from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
import tiktoken
from xai_gpt_shap_lima.roles import get_role_message

class ChatGptClient:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_history = []
        self.system_message = "You are an assistant designed to help the user understand SHAP results and explain them."
        self.console = Console()
        self.session = PromptSession()
        self.temperature = 0.7
        self.max_tokens = 200
        self.history_tokens = 4096

    def count_tokens(self, message):
        """
        Count the number of tokens in the message
        """
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(message['content']))
        
    def clean_chat_history(self, history_tokens=0):
        """
        Clean the chat history based on the maximum tokens
        """

        if history_tokens == 0:
            history_tokens = self.history_tokens

        total_tokens = sum([self.count_tokens(message) for message in self.chat_history])
        
        if len(self.chat_history) <4:
            self.custom_console_message("Chat history is too short to clean", color="red")
            return
        
        if total_tokens < history_tokens:
            return
        
        self.custom_console_message(f"Number of tokens: {total_tokens}. Cleaning the chat history based on the maximum tokens...", "red")


        first_three_messages = self.chat_history[:3] #system, initial prompt, chat response
        middle_messages = self.chat_history[3:-2]
        last_messages = self.chat_history[-2:]

        # Critical tokens = first_three_messages + last_messages
        critical_tokens = sum([self.count_tokens(message) for message in first_three_messages])
        critical_tokens += sum([self.count_tokens(message) for message in last_messages])


        if critical_tokens > history_tokens:
            self.custom_console_message(
                "Critical tokens exceed the maximum number of allowed tokens!",
                color="red"
            )
            raise ValueError("Critical tokens exceed the maximum number of allowed tokens!")
        
        
        while total_tokens > history_tokens and middle_messages:
            middle_messages.pop(0)
            total_tokens = (critical_tokens + sum([self.count_tokens(message) for message in middle_messages])) 
            
        self.chat_history = first_three_messages + middle_messages + last_messages



    def set_temperature(self, temperature):
        """
        Set the temperature for the model
        """
        self.temperature = temperature

    def set_tokens(self, max_tokens):
        """
        Set the maximum tokens for the model
        """
        self.max_tokens = max_tokens

    def get_user_input(self):
        """
        GET user input
        """
        try:
            user_input = self.session.prompt("(You): ")  # Interaktivni vnos
            if user_input.lower() in ["exit", "quit"]:
                self.console.print("[bold red]Exiting chat. Goodbye![/bold red]")
                exit()
            return user_input.strip()
        
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[bold red]Exiting chat. Goodbye![/bold red]")
            exit()

    def set_system_message(self, message):
        """
        Sets the system message to set the tone of the conversation
        """
        self.system_message = {"role": "system", "content": message}
        self.chat_history.insert(0, self.system_message)

    def send_initial_prompt(self, prompt, print_response=True, max_tokens=0, temperature=0):
        """
        Sends the initial prompt to the model
        """

        # If max_tokens is not set, use the default value
        if max_tokens == 0:
            max_tokens = self.max_tokens

        # If temperature is not set, use the default value
        if temperature == 0:
            temperature = self.temperature

        self.custom_console_message("Sending the initial SHAP results to the model...", "green")
        self.chat_history.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": answer})

        if print_response == True:
            self.console.print(
                Panel(
                    Markdown(answer), title="Assistant Response", border_style="blue"
                )
            )
        return answer

    def custom_console_message(self, message, color="white"):
        """
        Send a custom message to the console
        """
        self.console.print(f"[bold {color}]{message}[/bold {color}]")

    def interactive_chat(self):
        """
        Interactive chat with ChatGPT
        """
        self.console.print("[bold cyan]You can now interact with ChatGPT. Type your questions below![/bold cyan]")
        while True:
            try:
                #Get user input
                user_message = self.get_user_input()
                if not isinstance(user_message, str):
                    user_message = str(user_message)
                self.chat_history.append({"role": "user", "content": user_message})

                #Send input and stream the response
                self.stream_response()

                #Clean the chat history
                self.clean_chat_history()

            except (KeyboardInterrupt, EOFError):
                self.exit_chat()
                break

    def stream_response(self):
        """
        Stream the response from ChatGPT
        """
        self.console.print("[bold green]Streaming response from ChatGPT...[/bold green]")

        text = ""
        block = "█ "

        with Live(console=self.console, refresh_per_second=4) as live:
            for token in self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                temperature=self.temperature,
                stream=True,
            ):
                # Preveri, ali atribut "content" obstaja in ni None
                content = getattr(token.choices[0].delta, "content", None)
                if content:
                    text += content
                    markdown = Markdown(text + block)
                    live.update(markdown, refresh=True)

            live.update(
                Panel(
                    Markdown(text),
                    title="Assistant Response",
                    border_style="blue",
                )
            )

        self.chat_history.append({"role": "assistant", "content": text})
        return text
    
    def create_summary_and_message(self,shap_df, model, short_summary, choice_class, role):

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

    def choose_gpt_expertise_layer_interactive(self):

        roles_config = {
                "1": {"role": "beginner", "temperature": 1, "tokens": 300, "message": "Explain it to me like I'm a beginner"},
                "2": {"role": "student", "temperature": 0.8, "tokens": 200, "message": "Explain it to me like I'm a student"},
                "3": {"role": "analyst", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like I'm an analyst"},
                "4": {"role": "researcher", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like I'm a researcher"},
                "5": {"role": "executive_summary", "temperature": 0.8, "tokens": 150, "message": "Explain it to me like an executive summary"},
                "0": {"role": "pirate", "temperature": 0.8, "tokens": 200, "message": "Explain it to me like I'm a pirate"},
            }

        while True:

            self.custom_console_message("Please choose the ChatGPT expertise level you want to interact with:")
            self.custom_console_message("1. Beginner (Explain it to me like I'm a beginner)")
            self.custom_console_message("2. Student (Explain it to me like I'm a student)")
            self.custom_console_message("3. Analyst (Explain it to me like I'm an analyst)")
            self.custom_console_message("4. Researcher (Explain it to me like I'm a researcher)")
            self.custom_console_message("5. Executive Summary (Explain it to me like an executive summary)")
            self.custom_console_message("6. Exit")
            
            
            choice = self.get_user_input()

            if choice in roles_config:
                
                config = roles_config[choice]
                role = config["role"]
                self.set_temperature(config["temperature"])
                self.set_tokens(config["tokens"])
                self.custom_console_message(f"You chose the expertise level: {config['message']}", "yellow")
                
                # Set system msg
                try:
                    system_message = get_role_message(role)
                    self.set_system_message(system_message)
                    return role  
                except ValueError as e:
                    self.custom_console_message(f"[red]{e}[/red]")
            elif choice == "6":
                self.custom_console_message("Exiting chat. Goodbye!", "red")
                exit()
            else:
                self.custom_console_message("Invalid choice. Please try again.", "red")

    def set_gpt_expertise_layer(self, role=None):
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
                self.set_system_message(system_message)
                self.custom_console_message(f"Selected role: {role.capitalize()}", "green")
                return role
            except ValueError as e:
                self.custom_console_message(f"[red]{e}[/red]")
                raise e
        else:
            # if role not set use interacitve way
            return self.choose_gpt_expertise_layer_interactive()
        




