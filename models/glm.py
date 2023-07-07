import zhipuai
from langchain.llms.base import LLM
from typing import Optional, List
import os
# set the api key
zhipuai.api_key = os.getenv("ZHIPU_API_KEY")


class GLM(LLM):
    max_token = 2048
    temperature = 0.7
    top_p = 0.9

    history = []

    def __int__(self):
        super.__init__()

    def _llm_type(self):
        return "GLM"

    def _call(self, promt:str, stop:Optional[List[str]] = None):
        response = zhipuai.model_api.invoke(
            model="chatglm_130b",
            prompt=[
                     {"role": "user", "content": promt}
            ],
            temperature=self.temperature,
            top_p=0.9,
            incremental=True
        )

        if response["code"] != 200:
            return "模型返回错误"

        answer = response["data"]["choices"][0]["content"]
        #answer = answer.replace("\\n", "\n").rstrip('"').lstrip('"')

        return answer
