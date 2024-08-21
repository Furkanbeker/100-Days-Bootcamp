numbers = [1 ,2 ,3 ]

new_list = []

for n in numbers :
    add = n + 1
    new_list.append(add)

new_list = [ n*n for n in numbers]
print(new_list)

name = "Angela"
letters_list =  [letter for letter in name]
print(letters_list[0].upper())

listtt=[num*2 for num in range(1,5)]
print(listtt)

############################################

names = ["Alex" , "Beth" , "Caroline" , "Dave" ,  "Eleanor" , "Freddie"]
short_names = [name for name in names if len(name) < 5]
print(short_names)

long_names = [name.upper() for name in names if len(name) > 5]
print(long_names)

############################################
import random

names = ["Alex" , "Beth" , "Caroline" , "Dave" ,  "Eleanor" , "Freddie"]

student_scores = {student : random.randint(1,100) for student in names}
print(student_scores)

passed_students = {student:score for (student , score) in student_scores.items() if score > 60}
print (passed_students)


sentence = "What is the Airspeed Velocity of an Unladen Swallow?"
split=sentence.split()
result = {word:len(word) for word in split }
print ( result)

weather_c = {"Monday": 12, "Tuesday": 14, "Wednesday": 15, "Thursday": 14, "Friday": 21, "Saturday": 22, "Sunday": 24}

weather_f = {day:((temp_c * 9/5) + 32) for (day,temp_c) in weather_c.items()}

print(weather_f)
