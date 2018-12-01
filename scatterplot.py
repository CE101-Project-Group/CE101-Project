"""""
A program that charts the
relationship between house price
and the size of the house
created by Maryamu Zakariya
date created 30/11/18
modified 30/11/18
"""""

import pandas as pd
import matplotlib.pyplot as plt 


d = pd.read_csv(r"/Users/maryamu/Desktop/CSEE year 1  /dataforhousepricing.csv")
size = d['SQUARE FEET (SIZE)'] #xaxis
price  = d['PRICE'] #yaxis
plt.scatter(size, price, edgecolors='r')
plt.xlabel('House Size(ft square)')
plt.ylabel('House price')
plt.title('House Pricing Data')
plt.show() #shows the plot 

