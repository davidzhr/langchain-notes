# can not access this search engine from native

from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

print(search.run("Obama's first name?"))