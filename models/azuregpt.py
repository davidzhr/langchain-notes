
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from typing import Union
import os

model = AzureChatOpenAI(os.getenv("OPENAI_API_BASE"),
                        os.getenv("OPENAI_API_VERSION"),
                        os.getenv("CHATGPT_MODEL"),
                        os.getenv("AZURE_API_KEY"),
                       )


def chat(prompt: str) -> Union[str, None]:

    try:
        chat_hist = list()
        chat_hist.append(HumanMessage(content=f"{prompt}"))
        result = model(chat_hist)
        return result.content

    except Exception as e:

        return None
