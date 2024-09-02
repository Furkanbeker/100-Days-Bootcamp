#QUESTION I
##########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("********************************************************************************")

score = eval(input("Enter a grade value (0 to 100): "))

if score >= 95:
    grade = "A"

elif score >= 90:
    grade = "A-"

elif score >= 85:
    grade = "B+"

elif score >= 80:
    grade = "B"

elif score >= 75:
    grade = "B-"

elif score >= 70:
    grade = "C+"

elif score >= 65:
    grade = "C"

elif score >= 60:
    grade = "C-"

else:
    grade = "F"

print("The letter grade is ", grade)


##########################################################################################
#QUESTION II
##########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("********************************************************************************")


a=int(input("Enter a value."))
def calculate(b):
  total = sum(i for i in range(b) if(i % 3== 0))
  return total
print(calculate(a))



##########################################################################################
#QUESTION III
##########################################################################################

print("\n")
print("SOLUTION OF QUESTION III:")
print("********************************************************************************")

x,y = eval(input("Enter a point with two coordinates: "))

if -5<=x<=5 and -5/2<=y<=5/2:
    print("Point",(float(x),float(y)),"is in the rectangle")

else:
    print("Point",(float(x),float(y)),"is not in the rectangle")


##########################################################################################
#QUESTION IV
##########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("********************************************************************************")

print("a)")
print("\t")
for i in range(1,6):
    for j in  range(1,i+1):
        print(str(j), end=" ") 
    print()
print("\n")

print("b)")
print("\t")
a=6
for i in range(a+1, 0, -1):
    for j in range(1, i-1):
        print(str(j), end=" ")
    print("")
print("\n")

print("c)")
print("\t")
for i in range(1,6):
    for j in range(i):
        print(chr(64 + i), end=" ")        
    print()
print("\n")

print("d)")
print("\t")
for i in range(5):
    for j in range(i,5):
        print(" ", end=" ")
    for j in range(i):
        print("*", end=" ") 
    for j in range(i+1):
        print("*", end=" ") 
    print()

##########################################################################################
#QUESTION V
##########################################################################################
print("\n")
print("SOLUTION OF QUESTION V:")
print("********************************************************************************")

import math
print("This program computes value of sin(x) where x is in degrees and also computes the error  \n")

print("x \t sin(x) \t\t exact value \t\t error  \n")

for x in [ 45,60,90 ]:

    for n in [100,500,1000,10000]:

        sine = 0

        for i in range(n):

            sign = (-1)**i

            pi=22/7

            y=x*(pi/180)

            sine = sine + ((y**(2.0*i+1))/math.factorial(2*i+1))*sign
            break

        print(x,"\t", sine , "\t ",math.sin(x*(pi/180)),"\t",math.sin(x*(pi/180))-sine )

##########################################################################################
#

##########################################################################################
#QUESTION VI
##########################################################################################
print("\n")
print("SOLUTION OF QUESTION VI:")
print("********************************************************************************")

import turtle
import math


t=turtle.Turtle()
t.color("black")
t.penup()
t.goto(-200,0)
t.pendown()
t.goto(200,0)
t.right(135)
t.forward(15)
t.backward(15)
t.left(270)
t.forward(15)
t.penup()
t.goto(0,-100)
t.pendown()
t.right(45)
t.goto(0,100)
t.right(135)
t.forward(15)
t.backward(15)
t.left(270)
t.forward(15)
t.penup()
t.goto(-175,50)
t.pendown()
t.color("red")
for a in range(-175,176):
    t.goto(a, 50 * math.sin((a/100)*2*math.pi))

t.penup()
t.goto(-175,0)
t.pendown()
t.color("blue")
for a in range(-175,176):
    t.goto(a, 50 * math.cos((a/100)*2*math.pi))

t.penup()
t.goto(-100,-15)
t.pendown()
t.color("black")
t.write("-2\u03c0")

t.penup()
t.goto(100,-15)
t.pendown()
t.write("2\u03c0")

t.penup()
t.goto(500,0)

turtle.done()

##################################
