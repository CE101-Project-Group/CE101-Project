"""
Implements a house class which contains all the data about a single house
"""
from collections import OrderedDict
class House:
    def __init__(self, dataLoc, line):
        """
        Creates a house object
        dataLoc: string: a file with the data on all houses
        header is important
        line: int: the line on which the data about out particular house is located
        """
        #creates a dictionary that will contain all the bloody 80 variables and their names
        self.variables=OrderedDict()
        f=open(dataLoc)
        names=f.readline()
        namesList=names.split(',')
        for name in namesList:
            self.variables.update({name:None})
        i=0
        while i+1<line:
            f.readline()
            i=i+1
        else:
            values=f.readline()
        i=0
        valuesList=values.split(',')
        for item in self.variables:
            self.variables.update({item:valuesList[i]})
            i=i+1
        f.close
    def callVar(self, name):
        """
        Returns a data points about the house, whose name is passed in the arguements
        args: string: the name of desired data point
        """
        return self.variables[name]
place=House('test.csv', 2)
print(place.callVar('Id'))