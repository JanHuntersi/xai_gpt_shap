from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# TODO 
# Implementacija razreda za "nivo"
# tukaj imamo recimo 2-3 različne nivoje ki jih uporabnik lahko izbere:
# Nivo 1 "Explain like im five" mogoče z višjo temparaturo npr: 1.5
# Nivo 2 "Explain like im a student" z nižjo temparaturo npr: 0.5
# Nivo 3 "Explain like im a professor" z nižjo temparaturo npr: 0.1


# TODO razišči vstavitve


# TODO briši zgodovino da ne bo preveč dolga

# TODO ...

class ChatGptClient:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_history = []
        self.system_message = "You are a an asistant, designed to help the user understand SHAP results and explain them."
        self.console = Console()

    def get_user_input(self):
        """
        Pridobi vnos uporabnika za pogovor.
        """
        try:
            # Prikaže poziv za vnos uporabnika
            user_input = input("(You): ").strip()
            if user_input.lower() in ["exit", "quit"]:
                self.console.print("[bold red]Exiting chat. Goodbye![/bold red]")
                exit()
            return user_input
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[bold red]Chat končan s strani uporabnika.[/bold red]")
            exit()


    def set_system_message(self, message):
        """
        Sets the system message  to set the tone of the conversation.
        """
        self.system_message = {"role": "system", "content": message}
        self.chat_history.insert(0,self.system_message)



    def send_initial_prompt(self,prompt):
        """
        Sends the initial prompt to the model.
        """
        self.console.print(Markdown(f"Sending the initial shap results to the model..."))
        self.chat_history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history 
        )
        answer = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": answer})
        self.console.print(Markdown(f"**GPT Response:**\n\n{answer}"))
        return answer
    


    def custom_console_message(self,message):
        """
        Send a custom message to the console
        """
        self.console.print(Markdown(f"{message}"))



    def interactive_chat(self):
        """
        Interactive chat with ChatGPT
        """
        self.console.print("[bold cyan]You can now interact with ChatGPT. Type your questions below![/bold cyan]")
        while True:
            try:
                # Pridobi vnos uporabnika
                user_message = self.get_user_input()
                
                # Preverite, ali je vnos pravilen niz
                if not isinstance(user_message, str):
                    user_message = str(user_message)
                
                # Dodaj uporabniški vnos v zgodovino in pošlji
                self.chat_history.append({"role": "user", "content": user_message})
                self.stream_response(user_message)
            except (KeyboardInterrupt, EOFError):
                self.exit_chat()
                break


    def stream_response(self, prompt):
        """
        Pošlje uporabniški prompt ChatGPT-ju in prikaže streaming odgovor.
        """
        self.console.print("[bold green]Streaming response from ChatGPT...[/bold green]")

        text = ""
        block = "█ "

        with Live(console=self.console, refresh_per_second=4) as live:
            for token in self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,  # Pošlje zgodovino pogovora, ki že vsebuje uporabniški vnos
                stream=True,
            ):
                # Preveri, ali atribut "content" obstaja in ni None
                content = getattr(token.choices[0].delta, "content", None)
                if content:
                    text += content  
                    markdown = Markdown(text + block)
                    live.update(markdown, refresh=True)

            live.update(Markdown(text))

        self.chat_history.append({"role": "assistant", "content": str(text)})
        return text








