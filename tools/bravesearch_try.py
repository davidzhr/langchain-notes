from langchain.tools import BraveSearch

api_key = "BSAv1neIuQOsxqOyy0sEe_ie2zD_n_V"

tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})

print(tool.run("青岛海尔"))