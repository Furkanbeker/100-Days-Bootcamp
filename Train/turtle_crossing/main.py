import time
from turtle import Screen
from player import Player
from car_manager import CarManager
from scoreboard import Scoreboard

screen = Screen()
screen.setup(width=600, height=600)
screen.tracer(0)
screen.title("Turtle Crossing Game")


player = Player()
carmanager = CarManager()
scoreboard = Scoreboard()



screen.listen()
screen.onkey(player.go_up , "Up")
screen.onkey(player.go_down , "Down")




game_is_on = True
while game_is_on:
    time.sleep(0.1)
    screen.update()

    carmanager.create_cars()
    carmanager.move_cars()
    for car in carmanager.all_cars:
        if car.distance(player) < 20:
            game_is_on = False
            scoreboard.game_over()

    if player.is_at_finish_line():
        player.go_to_start()
        scoreboard.increase_level()



screen.exitonclick()





