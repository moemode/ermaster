import pickle
# Later, you can load the object back
with open('my_object.pkl', 'rb') as file:
    c = pickle.load(file)
    c.items()
    print(c)
    
#from openai.openai_object import OpenAIObject
