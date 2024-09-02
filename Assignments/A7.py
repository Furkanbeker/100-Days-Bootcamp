#!/usr/bin/env python
# coding: utf-8

# In[1]:

# QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")


# In[2]:


########################################################################################
# QUESTION I.I
########################################################################################

import numpy as np

# Create arrays of shape (9, 1) and (1, 9) using NumPy's arange function
row_array = np.arange(1, 10).reshape(9, 1)
col_array = np.arange(1, 10).reshape(1, 9)

# Multiply the two arrays to get the output array
out_array = row_array * col_array

# Print the output array
out_array


# In[3]:


########################################################################################
# QUESTION I.II
########################################################################################

import numpy as np

# Generate a 4x4 array of random floats between 0 and 1
rand_floats = np.random.random((4, 4))

# Scale the random floats to the desired range and convert to integers
arr = np.floor(rand_floats * 31) + 5

arr


# In[4]:


########################################################################################
# QUESTION I.III
########################################################################################

import numpy as np

def print_arrays(array_list):
  for array in array_list:
    print(array)

ones_array = np.ones((2, 3))
zeros_array = np.zeros((3, 3))
sevens_array = np.full((2, 5), 7)

array_list = [ones_array, zeros_array, sevens_array]
array_list


# In[5]:


########################################################################################
# QUESTION I.IV
########################################################################################

import numpy as np

# Create the first array containing even integers from 2 to 18
arr1 = np.linspace(2, 18, 9).reshape((3, 3)).astype(int)
arr1

# Create the second array containing integers from 9 to 1
arr2 = np.linspace(9, 1, 9).reshape((3, 3)).astype(int)
arr2

# Perform element-wise multiplication of the first array by the second
result = arr1 * arr2
arr1,arr2,result


# In[6]:


########################################################################################
# QUESTION I.V
########################################################################################

import numpy as np

# Create the input array
input_arr = np.arange(1, 16).reshape((3, 5))
input_arr


# In[7]:


# Select row 2
row_2 = input_arr[2]
row_2


# In[8]:


# Select column 4
col_4 = input_arr[:, 4]
col_4


# In[9]:


# Select the first two columns of rows 0 and 1
cols_01 = input_arr[0:2, 0:2]
cols_01


# In[10]:


# Select columns 2–4
cols_234 = input_arr[:, 2:5]
cols_234


# In[11]:


# Select the element that is in row 1 and column 4
elem_14 = input_arr[1, 4]
elem_14


# In[12]:


# Select all elements from rows 1 and 2 that are in columns 0, 2 and 4
rows_12_cols_024 = input_arr[1:3, [0, 2, 4]]
rows_12_cols_024


# In[13]:


########################################################################################
# QUESTION I.VI
########################################################################################

import numpy as np

# Create the input matrix
matrix = np.array([[10, 10, 90, 90],
                   [40, 40, 60, 60],
                   [55, 55, 65, 65],
                   [10, 30, 60, 90]])

# Define the statistics functions
def max_func(arr):
    return np.max(arr)

def min_func(arr):
    return np.min(arr)

def mean_func(arr):
    return np.mean(arr)

def std_func(arr):
    return np.std(arr)

# Apply the functions to the matrix along the second axis
max_values = np.apply_along_axis(max_func, axis=1, arr=matrix)
min_values = np.apply_along_axis(min_func, axis=1, arr=matrix)
mean_values = np.apply_along_axis(mean_func, axis=1, arr=matrix)
std_values = np.apply_along_axis(std_func, axis=1, arr=matrix)

print("Original array:", "\n" , matrix)
print("The maximum and the minimum values of the array along the second axis:", "\n" , max_values, min_values)
print("Mean values along second axis:", "\n" ,mean_values)
print("Standard deviation along second axis:", "\n" ,std_values)


# In[14]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")


# In[15]:


import pandas as pd

x = pd.Series([7,11,13,17])
x


# In[16]:


x1 = pd.Series(100.0,index=[0,1,2,3,4])
x1


# In[20]:


import numpy as np 
x2 = pd.Series(np.random.randint(0,100, size=20))
x2
x2.describe()


# In[21]:


temperatures= pd.Series([98.6,98.9,100.2,97.9],index=['Julie','Charlie','Sam','Andrea'])
temperatures


# In[22]:


x3 = pd.Series({'Julie':98.6 ,'Charlie':98.9,'Sam':100.2,'Andrea':97.9})
x3


# In[23]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")


# In[24]:


import pandas as pd
import numpy as np

data = {'Maxine': [23.5, 25.2, 24.8],
        'James': [26.1, 27.3, 26.8],
        'Amanda': [22.0, 23.5, 22.8]}

temperatures = pd.DataFrame(np.array([data[name] for name in data]).T, columns=data.keys())
temperatures


# In[25]:


temperatures = pd.DataFrame(data, index=['Morning', 'Afternoon', 'Evening'])
temperatures



# In[26]:


maxine_temps = temperatures['Maxine']
maxine_temps


# In[27]:


morning_temps = temperatures.loc['Morning']
morning_temps


# In[28]:


morning_evening_temps = temperatures.loc[['Morning', 'Evening']]
morning_evening_temps


# In[29]:


amanda_maxine_temps = temperatures[['Amanda', 'Maxine']]
amanda_maxine_temps


# In[30]:


amanda_maxine_morning_afternoon = temperatures.loc[['Morning', 'Afternoon'], ['Amanda', 'Maxine']]
amanda_maxine_morning_afternoon


# In[31]:


temps_desc = temperatures.describe()
temps_desc


# In[42]:


temperatures_transposed = temperatures.T
temperatures_transposed


# In[43]:


temperatures_sorted = temperatures.sort_index(axis=1)
temperatures_sorted


# In[32]:


########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A2
########################################################################################
########################################################################################
# QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("****************************")


# In[33]:


import pandas as pd

# Create a context manager for setting pandas display options
class display_options:
    def __init__(self, max_rows=500, max_columns=500):
        self.max_rows = max_rows
        self.max_columns = max_columns

    def __enter__(self):
        pd.set_option('display.max_rows', self.max_rows)
        pd.set_option('display.max_columns', self.max_columns)

    def __exit__(self, *args):
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')


# Read the CSV file
csv_file_path = r"C:\Users\mfb36\Desktop\csv-files-for-use\DEPARTMENTS.csv"

with display_options():
    departments = pd.read_csv(csv_file_path)
    
departments


# In[34]:


import pandas as pd

file_paths = {
    "employees": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv",
    "departments": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\DEPARTMENTS.csv",
    "job_history": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\JOB_HISTORY.csv",
    "jobs": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\JOBS.csv",
    "countries": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\COUNTRIES.csv",
    "regions": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\REGIONS.csv",
    "locations": "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\LOCATIONS.csv",
}

data_frames = {}

for name, path in file_paths.items():
    data_frames[name] = pd.read_csv(path)
    print(f"Number of records in {name}: ", data_frames[name].shape[0])


# In[35]:


import pandas as pd

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def filter_high_salary_employees(employees_df, salary_threshold):
    return employees_df[employees_df["salary"] > salary_threshold]

if __name__ == "__main__":
    file_path = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv"
    salary_threshold = 10000

    employees = read_csv_file(file_path)
    high_salary_employees = filter_high_salary_employees(employees, salary_threshold)

high_salary_employees


# In[36]:


import pandas as pd

def load_employees(file_path):
    employees = pd.read_csv(file_path)
    employees["commission_pct"].fillna(0, inplace=True)
    return employees

file_path = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv"
employees_data = load_employees(file_path)
employees_data


# In[37]:


import pandas as pd

# Define a function to filter employees by department ID
def filter_by_dept_id(employee_data, dept_ids):
    return employee_data["department_id"].isin(dept_ids)

# Read employee data from CSV
employee_data = pd.read_csv("C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv")

# Filter employees by department ID
department_ids = [30, 50, 80]
filtered_employees = employee_data.loc[filter_by_dept_id(employee_data, department_ids), ["first_name", "last_name", "salary", "department_id"]]

# Print the filtered employees
filtered_employees


# In[38]:


import pandas as pd

def read_csv_file(file_path):
    return pd.read_csv(file_path)

employees_file = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv"
departments_file = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\DEPARTMENTS.csv"

employees_data = read_csv_file(employees_file)
departments_data = read_csv_file(departments_file)

emp_dept = pd.merge(employees_data, departments_data, on="department_id")
emp_dept


# In[39]:


import pandas as pd

employees_file = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\EMPLOYEES.csv"
departments_file = "C:\\Users\\mfb36\\Desktop\\csv-files-for-use\\DEPARTMENTS.csv"

with open(employees_file, "r") as emp_file, open(departments_file, "r") as dept_file:
    employees = pd.read_csv(emp_file)
    departments = pd.read_csv(dept_file)

emp_dept = pd.merge(employees, departments, on="department_id")

salary_stats = (
    emp_dept
    .groupby("department_name")
    .agg({"salary": ["min", "max", "mean"]})
)

salary_stats


# In[40]:


import pandas as pd

# Sample data for locations and emp_dept
locations = pd.DataFrame({'location_id': [1, 2, 3], 'country_id': ['US', 'UK', 'AU'], 'city': ['New York', 'London', 'Sydney']})
emp_dept = pd.DataFrame({'location_id': [1, 1, 2, 2, 3], 'salary': [6000, 4500, 12000, 8000, 7000]})

loc_emp_dept = locations.merge(emp_dept, on='location_id')
salary_ranges = [0, 5000, 10000, 15000, 25000]
loc_emp_dept['salary_range'] = pd.cut(loc_emp_dept['salary'], bins=salary_ranges)
grouped_salaries = loc_emp_dept.groupby(['country_id', 'city', 'salary_range']).mean()['salary']
grouped_salaries = grouped_salaries.unstack('salary_range').fillna(0)
grouped_salaries

