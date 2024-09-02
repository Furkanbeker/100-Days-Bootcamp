#QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")

print('A program to display the following table')
print('a',                     'a^3',                     'a^4')
print('1',                      1**3,                      1**4)
print('2',                      2**3,                      2**4)
print('3',                      3**3,                      3**4)
print('4',                      4**3,                      4**4)

########################################################################################
#QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")

x = 4*(1-1/3+1/5-1/7+1/9-1/11)
y = x+4*(1/13-1/15)
print(x)
print(y)
########################################################################################
#QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")

abcdef = eval(input("Enter a six-digit integer"))
a=(abcdef)//100000
b=(abcdef-a*100000)//10000
c=(abcdef-a*100000-b*10000)//1000
d=(abcdef-a*100000-b*10000-c*1000)//100
e=(abcdef-a*100000-b*10000-c*1000-d*100)//10
f=(abcdef-a*100000-b*10000-c*1000-d*100-e*10)

print("", f)
print("", e)
print("", d)
print("", c)
print("", b)
print("", a)

########################################################################################
#QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IVa:")
print("****************************")

numbers = input("Enter a Number to Convert to ASCII code:")
s1=int(numbers)
print("Character of ASCII value", numbers, "is: ", chr(s1))

print("\n")
print("SOLUTION OF QUESTION IVb:")
print("****************************")
import turtle

t = turtle.Turtle()
wn = turtle.Screen()

t.speed(1)
t.color('black')
style = ('Ahorani', 40, 'italic')
t.fillcolor('black')
t.up()
t.goto(20, 0)
t.down()
t.begin_fill()
t.write(u'\u062C', font=style)
t.end_fill()

t.up()
t.goto(80, 0)
t.down()
t.color('black')
t.fillcolor('black')
t.begin_fill()
t.write(u'\u0634', font=style)
t.end_fill()

t.up()
t.goto(0, -100)
t.down()
t.color('black')
t.fillcolor('black')
t.begin_fill()
t.write(u'\u0394', font=style)
t.end_fill()

t.up()
t.goto(50, -100)
t.down()
t.color('black')
t.fillcolor('black')
t.begin_fill()
t.write(u'\u03B4', font=style)
t.end_fill()

t.up()
t.goto(100, -100)
t.down()
t.color('black')
t.fillcolor('black')
t.begin_fill()
t.write(u'\u03D5', font=style)
t.end_fill()

wn.reset()
t.hideturtle()

########################################################################################
#QUESTION V
########################################################################################
print("\n")
print("SOLUTION OF QUESTION Va:")
print("****************************")

import turtle
t=turtle.Turtle()
t.speed(2)
t.pensize(2)
wn=turtle.Screen()
wn.bgcolor('light blue')
t.up()
t.goto(150,0)
t.down()
t.goto(-150,0)
t.up()
t.goto(0,150)
t.down()
t.goto(0,-150)
t.up()
t.goto(0,-20)
t.down()
t.circle(20)
t.color('black')
style = ('Arial', 10, 'italic')
t.fillcolor('black')
t.begin_fill()
t.penup()
t.goto(170,-5)
t.pendown()
t.write('East' ,font=style)
t.end_fill()

t.begin_fill()
t.penup()
t.goto(-185,-5)
t.pendown()
t.write('West' ,font=style)
t.end_fill()

t.begin_fill()
t.penup()
t.goto(-10,160)
t.pendown()
t.write('North' ,font=style)
t.end_fill()

t.begin_fill()
t.penup()
t.goto(-10,-170)
t.pendown()
t.write('South' ,font=style)
t.end_fill()

t.hideturtle()

wn.reset()


print("\n")
print("SOLUTION OF QUESTION Vb:")
print("****************************")


import turtle
t=turtle.Turtle()
wn=turtle.Screen()
wn.bgcolor('white')
t.circle(150)
t.penup()
t.goto(-100,90)
t.pendown()
t.goto(0,20)
t.goto(100,90)
t.penup()
t.goto(60,100)
t.pendown()
t.goto(0,180)
t.goto(-60,100)
t.penup()
t.goto(-80,180)
t.pendown()
t.begin_fill()
t.color('black')
t.circle(20)
t.end_fill()
t.penup()
t.goto(80,180)
t.pendown()
t.begin_fill()
t.color('black')
t.circle(20)
t.end_fill()

t.hideturtle()

