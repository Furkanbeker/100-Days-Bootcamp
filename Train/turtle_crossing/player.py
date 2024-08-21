from turtle import Turtle

STARTING_POSITION = (0, -280)
MOVE_DISTANCE = 10
FINISH_LINE_Y = 280

UP = 90
DOWN = 270

class Player(Turtle):
    def __init__(self):
        super().__init__()
        self.shape("turtle")
        self.color("black")
        self.penup()
        self.goto(STARTING_POSITION)
        self.setheading(90)

    def go_to_start(self):
        self.goto(STARTING_POSITION)

    def go_up(self):
        self.forward(MOVE_DISTANCE)

    def go_down(self):
        self.forward(-MOVE_DISTANCE)

    def is_at_finish_line(self):
        if self.ycor() > (FINISH_LINE_Y):
            return True
        else:
            return False


