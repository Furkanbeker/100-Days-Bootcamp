import turtle as t
import random

tim = t.Turtle()
t.colormode(255)

def random_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    random_color = ( r , g , b)
    return random_color

tim.speed(500)
tim.pensize(25)
tim.up()
tim.goto( -250 , -250)
tim.down()


def go():
    for i in range (7):
        if i <= 4:

            tim.color(random_color())
            tim.forward(1)
            tim.up()
            tim.forward(100)
            tim.down()
            tim.forward(1)
        else:
            return

for y in range (-250 , 251):
    if y % 75 == 0:
        tim.up()
        tim.goto( -250 , y )
        tim.down()
        go()

    else:
        print("")

tim.hideturtle()
screen = Screen()
screen.exitonclick()
