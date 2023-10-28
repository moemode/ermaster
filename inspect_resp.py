import pickle

# Later, you can load the object back
with open("my_object.pkl", "rb") as file:
    c = pickle.load(file)

with open("completions.pkl", "rb") as file:
    cs = pickle.load(file)

print(c)
# from openai.openai_object import OpenAIObject
