import matplotlib.pyplot as plt
import numpy as np
 
all_data = [125000,50000,100000,75000]
 
fig = plt.figure(figsize=(8,6))
 
plt.boxplot(all_data,
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            vert=True)   # vertical box aligmnent
 
plt.xticks([y+1 for y in range(1)], ['house price'])
plt.xlabel('price range')
t = plt.title('Box plot')
plt.show() 
