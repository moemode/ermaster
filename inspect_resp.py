import pickle
# Later, you can load the object back
with open('my_object.pkl', 'rb') as file:
    c = pickle.load(file)
    print(c)