import turtle as t
from turtle import Screen
import random

t.colormode(255)

tim = t.Turtle()
tim.speed(150)

def random_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    random_color = ( r , g , b)
    return random_color

for i in range (75):
    tim.color(random_color())
    tim.circle(100)
    tim.left(5)

tim.hideturtle()
screen = Screen()
screen.exitonclick()