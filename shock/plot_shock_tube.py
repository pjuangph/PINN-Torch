import pickle
import matplotlib.pyplot as plt 

with open('history.pickle', 'rb') as handle:
    results = pickle.load(handle)

temp = list(filter(lambda item: item['t'] == 0.2, results['history']))
for t in temp:
    v = t['history']
    T = v[:,0]
    P = v[0,:]*287*T
print("Results loaded")



