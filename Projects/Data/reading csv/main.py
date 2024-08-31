# print(data["temp"])
# print(data.condition)
#
# temp_list = data["temp"].to_list()
# print(temp_list)
#
# # print(f"average : {sum(temp_list) / len(temp_list)}")
# print(data["temp"].mean())
# print(data["temp"].max())
#
# print(data[data.day == "Monday"])
# print(data[data.temp == data["temp"].max()])
#
# monday = data[data.day == "Monday"]
# x = monday["temp"]
# print (x+33.8)
#

import pandas
data = pandas.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data_20240807.csv")
(grey_squirrels) = len(data[data[ "Primary Fur Color" ] == "Gray"])
(red_squirrels) = len(data[data[ "Primary Fur Color" ] == "Cinnamon"])
(black_squirrels) = len(data[data[ "Primary Fur Color" ] == "Black"])

print(grey_squirrels)
print(red_squirrels)
print(black_squirrels)

data_dict= {
    "Fur Color": ["Gray","Cinnamon","Black"],
    "Count": [grey_squirrels , red_squirrels , black_squirrels]
}

df = pandas.DataFrame(data_dict)
df.to_csv("Squirrel_count.csv")

