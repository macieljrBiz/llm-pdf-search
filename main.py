import os
import chainlit as cl
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

# load_dotenv()

# Load environment variables
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = os.getenv("AOAI_KEY")
AOAI_MODEL = os.getenv("AOAI_MODEL")
AOAI_VERSION = os.getenv("AOAI_API_VERSION")

AOAI_CLIENT = AzureOpenAI(
    api_version=AOAI_VERSION,
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY
)

LLM_Context = []

def load_system_message(markdown_content=None):
    """
    This function loads the system message from the system_prompt.txt file.
    Returns:
        The system message.
    """
    system_message = ""
    with open("system_prompt.txt", "r", encoding="utf-8") as file:
        system_message = file.read()
    # Substitute the system_message text "{{MARKDOWN_CONTENT}}" placeholder with the actual content
    if markdown_content:
        system_message = system_message.replace("{{MARKDOWN_CONTENT}}", markdown_content)
    else:
        system_message = system_message.replace("{{MARKDOWN_CONTENT}}", "")
    return system_message

def load_markdown_content():
    """
    This function process a PDF file and returns the content as markdown.
    Returns:
        The content of the PDF file as markdown.
    """
    markdown = """
        | Fruit Name | Vitamins         | Curiosity                                      |
        |------------|------------------|------------------------------------------------|
        | Apple      | C, B6            | Apples float in water because they are 25% air.|
        | Banana     | B6, C            | Bananas are berries, but strawberries are not. |
        | Orange     | C                | Oranges are a hybrid of pomelo and mandarin.   |
        | Kiwi       | C, K, E          | Kiwi has more vitamin C than an orange.        |
        | Strawberry | C, B9            | Strawberries are the only fruit with seeds on the outside. |
        """
    return markdown

def update_context(message, role="user"):
    """
    This function updates the context (last 100 entries) with the role's message.
    Args:
        message: The entry message.
        role: The role of entry (user, assistant or system).
    Returns:
        The context with the updated message.
    """
    global LLM_Context
    if len(LLM_Context) == 0:
        LLM_Context.append({"role": "system", "content": load_system_message(load_markdown_content())})
    LLM_Context.append({"role": role, "content": message})
    if len(LLM_Context) > 100:
        LLM_Context.pop(0)
    return LLM_Context

@cl.step(type="llm")
async def llm(message: str):
    """
    This function represents the LLM (AOAI) tool that will process and answer to the message.
    Args:
        message: The entry message.
    Returns:
        The completion from the LLM.
    """
    global AOAI_CLIENT
    global AOAI_MODEL
    global LLM_Context
    # Update the context with the user's message
    LLM_Context = update_context(message)
    try:
        response = AOAI_CLIENT.chat.completions.create(
            messages=LLM_Context,
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=AOAI_MODEL
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request. Please try again later."


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.
    Args:
        message: The user's message.
    Returns:
        None.
    """
    # Call the tool
    tool_res = await llm(message.content)
    await cl.Message(content=tool_res).send()
    # message = f'Your message was: \n{message.content}. \n\nReceived!'
    # await cl.Message(content=message).send()  # this will be sent as the final answer

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)