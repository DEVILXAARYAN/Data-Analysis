import pandas as pd
import numpy as np
import time as t
import random
import urllib.request
from urllib.request import urlretrieve
import os


'''

# NUMPY

a = np.array([73, 67, 43])
b = np.array([0.5, 0.2, 0.3])
c = np.dot(a, b)
print("Using .dot = ", c)
d = (a * b).sum()
print("Using * and sum() = ",d)
'''


'''
print("FOR LOOP")
start1 = t.time()
arr1 = list(range(1000000))
arr2 = list(range(1000000, 2000000))

arr1_np = np.array(arr1)
arr2_np = np.array(arr2)

result = 0
for x1, x2 in zip(arr1, arr2):
    result += x1 * x2
print(result)
end1 = t.time()
elapsed1 = end1 - start1
print(f"Processing time: {elapsed1:.4f} seconds")

print("USING DOT")
start2 = t.time()
c = np.dot(arr1_np, arr2_np)
print(c)
end2 = t.time()
elapsed2 = end2 - start2
print(f"Processing time: {elapsed2:.4f} seconds")'
'''


'''
arr =  np.array([23, 43, 63])
arr1 =  np.array([0.23, 0.43, 0.63])
arr2 =  np.array(["0.23", "0.43", "0.63"])
print(arr.dtype, arr1.dtype, arr2.dtype)                                # -> Type of array
'''


'''
# DIMENSION OF ARRAY

arr1 = np.array([25, 34, 52])
print(arr1.shape)                                                       # -> One Dimensional Array

arr2 = np.array([[25, 34, 52],
                [32, 12, 5]])

print(arr2.shape)                                                       # -> Two Dimensional Array

arr3 = np.array([
                [[25, 34, 52],
                [32, 12, 5]],

                [[13, 43, 27],
                [6, 1, 3]]])

print(arr3.shape)                                                       # -> Three Dimensional Array'
'''


'''
# MATRIX MULTIPLICATION

a = np.array([[1, 3, 4],
              [1, 4, 2]])

b = np.array([1, 3, 4])

c = np.matmul(a, b)
d = a @ b
print(c)
print(d)
'''


'''
# GET THE DATA FROM CLIMATE_RESULTS.TXT

urllib.request.urlretrieve(
    'https://hub.jovian.ml/wp-content/uploads/2020/08/climate.csv',
    'climate.txt'
)
climate_data = np.genfromtxt('climate_results.txt', delimiter=',', skip_header=1)
print(climate_data)
print(climate_data.shape)
'''


'''
x = np.zeros((2, 3))
y = np.zeros((2, 3), dtype=int)
z = np.ones((4, 5, 3))
print(x)
print(y)
print(z)
'''


'''
a = np.arange(10)
a = np.arange(5, 10)
a = np.arange(1, 2, 0.1)
a = np.linspace(1., 5., 6)
a = np.full((2, 3), 8 )
a = np.eye(5)
a = np.eye(2, 5, 1)
a = np.random(5, 5)
print(a)
'''


'''
# ARRAY INDEXING

arr3 = np.array([
                [[25, 34, 52],
                [32, 12, 5]],

                [[13, 43, 27],
                [6, 1, 3]],
                
                [[52, 43, 25],
                [23, 21, 15]],
                ])

print(arr3.shape)   
print(arr3[1])   
print(arr3[1, 1])   
print(arr3[1, 1, 2])   
print(arr3[1:, 0:1, :2])   
'''


'''
# ARANGE WITH SHAPE AND RESHAPE

print(np.arange(10, 90, 3))
print(np.arange(10, 90, 3).shape)
print(np.arange(10, 90, 3).reshape(3, 3, 3))
'''


'''
# CONCATENATE WITH AXIS = 1

arr1 = [[25, 34, 52],
        [32, 12, 5]]
arr2 = [[13, 43, 27],
        [6, 1, 3]]
print(np.concatenate((arr1, arr2), axis = 1))


# CONCATENATE WITH AXIS = 0

arr3 = [[25, 34, 52],
        [32, 12, 5]]
arr4 = [[13, 43, 27],
        [6, 1, 3]]
print(np.concatenate((arr3, arr4), axis = 0))
'''


'''
# OS (OPERATING SYSTEM)

print(os.getcwd())                                # Get the Working Directory with os.getcwd() 
print(help(os.listdir))
print(os.listdir('.'))                            # Get the list of the relative path
print(os.listdir('pandas/Scripts'))               # Get the list of the absolute path
print(os.makedirs('./data', exist_ok=True))       # Make a new Directory 
print(os.makedirs('data', exist_ok=True))         # Make a new Directory

print('data' in os.listdir('.'))                  # Check the List in the Directory
print(os.listdir('data'))                         # Check the List in the Data Directory

url1 = 'https://gist.github.com/8de7b03f241b787042be1a1e4afd91da.git'
url2 = 'https://gist.github.com/aakashns/8de7b03f241b787042be1a1e4afd91da#file-loans2-txt'
url3 = 'https://gist.github.com/aakashns/8de7b03f241b787042be1a1e4afd91da#file-loans3-txt'

urllib.request.urlretrieve(url1, 'data/loans1.txt')
urllib.request.urlretrieve(url2, 'data/loans2.txt')
urllib.request.urlretrieve(url3, 'data/loans3.txt')
print(os.listdir('data')) 

file1 = open('data/loans1.txt', mode='r')           # Open a file from directory with read mode
file1_contents = file1.read()                       # Read the file 
print(file1_contents)
file1.close
print(file1_contents)

with open('data/loans2.txt', 'r') as file2:
    file2_contents = file2.read()
    print(file2_contents)

'''
with open('data/loans3.txt', 'r') as file3:
    file3_line = file3.readlines()                  # gives the number of lines
    # print(file3_line)
    # print(file3_line[0].strip())                  # strip() removed whitespace or remove newline character (\n)
    # print(file3_line[1])


'''
# READING FROM AND WRITING TO FILES USING PYTHON

# FOR HEADER

print("Header line")
def parse_header(header_line):
    return header_line.strip().split(',')
print(file3_line[0])

headers = parse_header(file3_line[0])
print(headers)

# FOR DATA

print("Data line")
def parse_values(data_line):
    values = []                                     # List
    for item in data_line.strip().split(','):
        if item == '':                              # if values has an empty string
            values.append(0.0)
        else:
            values.append(float(item))
    return values

print(file3_line[1])
print(file3_line[2])

print(parse_values(file3_line[1]))
print(parse_values(file3_line[2]))

# CREATING A DICTIONARY

def create_item_dict(values, headers):
    result = {}                                     # Dictionary
    for value, header in zip(values, headers):
        result[header] = value
    return result 

for item in zip([1, 2, 3], ['a', 'b', 'c']):      # for the understanding purpose of zip()
    print(item)

print(file3_line[1])
values1 = parse_values(file3_line[1])
print(create_item_dict(values1, headers))

print()

print(file3_line[2])
values1 = parse_values(file3_line[2])
print(create_item_dict(values1, headers))

# READ_CSV FUNCTION

def read_csv(path):
    result = []
    with open(path, 'r') as f:
        lines = f.readline()
        headers = parse_header(lines[0])
        for data_line in lines[1:]:
            values = parse_values(data_line)
            item_dict = create_item_dict(values, headers)
            result.append(item_dict)
    return result

a = read_csv('data/loans3.txt')
print(a)
''' 


'''

# PANDAS
'''

urlretrieve('https://github.com/RamiKrispin/covid19Italy/blob/master/csv/italy_total.csv')
covid_df = pd.read_csv('italy_total.csv')
print(covid_df)