from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
import tiktoken


# TODO razišči vstavitve


#TODO: Implement the logic to clean the chat history based on the maximum tokens


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
        
    def clean_chat_history(self, max_tokens):
        """
        Clean the chat history based on the maximum tokens
        """
        #TODO: Implement the logic to clean the chat history based on the maximum tokens

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

    def send_initial_prompt(self, prompt, max_tokens=0, temperature=0):
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
                user_message = self.get_user_input()

                if not isinstance(user_message, str):
                    user_message = str(user_message)

                self.chat_history.append({"role": "user", "content": user_message})

                self.stream_response()

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




