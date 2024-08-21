import turtle
from turtle import Turtle, Screen
import random

is_race_on = False

screen = Screen()
screen.setup(500 , 400)

user_bet = screen.textinput( title= "Make your bet" , prompt="Which turtle will win the race? Enter a color: ")
colors = ["red" , "orange" , "yellow" , "green" , "blue" , "purple"]
y_positions=[-70 , -40 , -10 , 20 , 50 , 80 ]
all_turtles = []

for turtle_index in range(0,6):
    tim = Turtle(shape="turtle")
    tim.color(colors[turtle_index])
    tim.up()
    tim.goto(x=-225 , y= y_positions[turtle_index])
    all_turtles.append(tim)

if user_bet:
    is_race_on = True

while is_race_on:

    for turtle in all_turtles:
        if turtle.xcor() > 230:
            print(turtle.color())
            is_race_on = False
            if turtle.pencolor() == user_bet:
                print("You won!")
            else:
                print(f"Sorry, you lost. Winner is: {turtle.pencolor()}")
        else:
            random_distance = random.randint(0 , 10)
            turtle.forward(random_distance)
