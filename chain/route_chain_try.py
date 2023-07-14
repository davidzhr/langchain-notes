
# from https://zhuanlan.zhihu.com/p/642971042

import time

import warnings
warnings.filterwarnings("ignore")


lawyer_template = """ 你是一个法律顾问，你需要根据用户提出的问题给出专业的法律意见，如果你不知道，请说"我不知道"，请不要提供超出文本范围外的内容，也不要自创内容。


用户提问：
{input}
"""


sales_template = """ 你是一个销售顾问，你需要为用户输入的商品进行介绍，你需要提供商品基本信息，以及其使用方式和保修条款。


用户输入的商品：
{input}
"""


english_teacher_template ="""你是一个英语老师，用户输入的中文词汇，你需要提供对应的英文单词，包括单词词性，对应的词组和造句。


用户输入的中文词汇：
{input}
"""


poet_template=""" 你是一个诗人，你需要根据用户输入的主题作诗。


用户输入的主题:
{input}
"""


prompt_infos = [
    {
        "name": "lawyer",
        "description": "咨询法律相关问题时很专业",
        "prompt_template": lawyer_template,
    },
    {
        "name": "sales",
        "description": "咨询商品信息时很专业",
        "prompt_template": sales_template,
    },
    {
        "name": "english teacher",
        "description": "能够很好地解答英语问题",
        "prompt_template": english_teacher_template,
    },
    {
        "name": "poet",
        "description": "作诗很专业",
        "prompt_template": poet_template,
    },
]


#创建一个list储存对应的子链名称和描述
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]


#把描述连接成一个str
destinations_str = "\n".join(destinations)


from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

from langchain.chains.router.llm_router import RouterOutputParser
from langchain.prompts import PromptTemplate

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)


from dotenv import load_dotenv
from langchain.chains.router.llm_router import LLMRouterChain
from langchain.chains.router import MultiPromptChain

from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationChain



import os
import openai

# load_dotnev will parse the .env file and set the variables in environment automatically.
load_dotenv()

# set the env variables for openai
#openai.api_key = os.getenv("AZURE_API_KEY")
#openai.api_version = os.getenv("OPENAI_API_VERSION")
#openai.api_base = os.getenv("OPENAI_API_BASE")
#openai.api_type = "azure"


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")


llm = AzureOpenAI(deployment_name=os.getenv("CHATGPT_MODEL"))
route_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

# 创建一个候选链，包含所有的下游子链
candadite_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    candadite_chains[name] = chain

# 生成默认链
default_chain = ConversationChain(llm=llm, output_key="text", verbose=True)


chain = MultiPromptChain(
    router_chain=route_chain,
    destination_chains=candadite_chains,
    default_chain=default_chain,
    verbose=True,
)


start = time.time()
print(chain.run("公司解约触发法律吗"))
end = time.time()
print(f'consume: {end-start}')