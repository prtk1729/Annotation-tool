# Python program to explain os.makedirs() method 

# importing os module 
import os
import json
from datetime import date


today = date.today()
print("Enter your name_initials: ")
name_initials = raw_input()
print(name_initials)
day, month, year = today.day, today.month, today.year

# def create_file(today, name_initials):
#     day, month, year = today.day, today.month, today.year
#     path = './StatsIO/{}/{}_{}_{}'.format(name_initials,day, month, year)
#     # os.makedirs(path)
#     try:
#         os.makedirs(path)
#         print("Directory created successfully" )
#     except OSError as error:
#         print("Directory already present")

# create_file(today, str(name_initials))

# print('After stmts')

path = './StatsIO/{}/{}_{}_{}'.format(name_initials,day, month, year)
file_name = "time_logs.json"
print(os.path.join(path, file_name)) #./StatsIO/pp/22_12_2020/time_logs.json