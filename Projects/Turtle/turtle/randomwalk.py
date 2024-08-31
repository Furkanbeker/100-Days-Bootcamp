from turtle import Turtle , Screen
import random

t = Turtle()
colours = ["CornflowerBlue", "DarkOrchid", "IndianRed", "DeepSkyBlue", "LightSeaGreen", "wheat", "SlateGray", "SeaGreen"]

t.speed(150)
t.pensize(8)

game_on = True

while game_on:
    t.color(random.choice(colours))
    x = random.randint(0,3)
    if x == 0:
        t.right(90)
    elif x == 1:
        t.left(90)
    elif x == 2:
        t.forward(25)
    elif x == 3:
        t.backward(25)
    else :
        game_on = False



