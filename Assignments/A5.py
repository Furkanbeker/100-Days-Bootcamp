########################################################################################
#Name:         Mustaa Furkan BEKER
#Student ID:   61210007
#Department:   Electrical and Electronics Engineering
#Assignment ID: A5
########################################################################################
########################################################################################
#QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")

# python code
import random
import sys

with open('inputs.txt', 'r') as f:
    lines = f.readlines()


def readMatrix(numberOfRows, numberOfColumns, file):
    matrix = []  # Create an empty list
    for row in range(numberOfRows):
        matrix.append([])  # Add an empty new row
        line = file.readline()
        rowdata = [int(x) for x in line.split(' ')]
        for column in range(numberOfColumns):
            matrix[row].append(rowdata[column])
    return matrix


def printMatrix(matrix):
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            print(format(matrix[row][column], "5d"), end=" ")
        print()  # Print a new line


def fillMatrixRandomly(numberOfRows, numberOfColumns):
    matrix = []  # Create an empty list
    for row in range(numberOfRows):
        matrix.append([])  # Add an empty new row
        for column in range(numberOfColumns):
            matrix[row].append(random.randint(0, 99))
    return matrix


def generateZeroMatrix(numberOfRows, numberOfColumns):
    matrix = [[0 for i in range(numberOfRows)] for j in range(numberOfColumns)]
    return matrix


def addMatrix(A, B):
    C = generateZeroMatrix(len(A), len(A[0]))
    for row in range(len(A)):
        for column in range(len(A[row])):
            C[row][column] = A[row][column] + B[row][column]

    return C


def subtractMatrix(A, B):
    C = generateZeroMatrix(len(A), len(A[0]))
    for row in range(len(A)):
        for column in range(len(A[row])):
            C[row][column] = A[row][column] + B[row][column]
    return C


def maxOfElements(A):
    max_val = A[0][0]
    for i in range(len(A)):
        for j in range(len(A[i])):
            max_val = max(max_val, A[i][j])
    return max_val


def transpose(A):
    B = generateZeroMatrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            B[i][j] = A[j][i]
    return B


def multiplyMatrix(A, B):
    C = generateZeroMatrix(len(A), len(B[0]))
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C


print("Reading data from inputs.txt file in current directory")
f = open("inputs.txt", "r")

line = f.readline()
numberOfRows, numberOfColumns = [int(x) for x in line.split(' ')]
A = readMatrix(numberOfRows, numberOfColumns, f)
print(" **** Matrix A **** ")
printMatrix(A)

line = f.readline()
numberOfRows, numberOfColumns = [int(x) for x in line.split(' ')]
B = readMatrix(numberOfRows, numberOfColumns, f)
print(" **** Matrix B **** ")
printMatrix(B)
line = f.readline()
numberOfRows, numberOfColumns = [int(x) for x in line.split(' ')]
C = readMatrix(numberOfRows, numberOfColumns, f)
print(" **** Matrix C **** ")
printMatrix(C)

D = fillMatrixRandomly(numberOfRows, numberOfColumns)
print(" **** Matrix D **** ")
printMatrix(D)

print(" *** Computing S = (A+B) * Transpose(C) + D) - A *** ")

# T1 = A + B
T1 = addMatrix(A, B)

print(" **** MatriX T1 = (A+B) ****")
printMatrix(T1)
T2 = transpose(C)
print("**** Matrix T2 = Transpose(C) ****")
printMatrix(T2)
T3 = multiplyMatrix(T1, T2)
print("**** Matrix T3 =(A+B) * transpose(C) ****")
printMatrix(T3)

T4 = addMatrix(T3, D)
print("**** Matrix T4 =(A+B) * transpose(C)+ D ****")
printMatrix(T4)

S = subtractMatrix(T4, A)
print("**** Matrix S =(A+B) * transpose(C) + D - A ****")
printMatrix(S)
max_val = maxOfElements(S)
print("Maximum Element in S = {} ".format(max_val))

########################################################################################
#QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")


from math import sqrt

class GeometricObject:
    def __init__(self, color = "white", filled = False):
        self.__color = color
        self.__filled = filled

    def getColor(self):
        return self.__color

    def setColor(self, color):
        self.__color = color

    def isFilled(self):
        return self.__filled

    def setFilled(self, filled):
        self.__filled = filled

    def __str__(self):
        return "color: " + self.__color + \
            " and filled: " + str(self.__filled)

class Triangle(GeometricObject):
    def __init__(self, side1 = 1.0, side2 = 1.0, side3 = 1.0, color = "white", filled = False):
        super().__init__(color, filled)
        self.__side1 = side1
        self.__side2 = side2
        self.__side3 = side3

        if self.__side1 <= 0 or self.__side2 <= 0 or self.__side3 <= 0:
            raise RuntimeError("Sides of a triangle must be positive.")

    def getSide1(self):
        return self.__side1

    def setSide1(self, side1):
        self.__side1 = side1

    def getSide2(self):
        return self.__side2

    def setSide2(self, side2):
        self.__side2 = side2

    def getSide3(self):
        return self.__side3

    def setSide3(self, side3):
        self.__side3 = side3

    def getPerimeter(self):
        return self.__side1 + self.__side2 + self.__side3

    def getArea(self):
        s = self.getPerimeter() / 2
        return sqrt(s * (s - self.__side1) * (s - self.__side2) * (s - self.__side3))

    def __eq__(self, other):
        return (self.__side1 == other.__side1 and self.__side2 == other.__side2 and self.__side3 == other.__side3) or \
               (self.__side1 == other.__side1 and self.__side2 == other.__side3 and self.__side3 == other.__side2) or \
               (self.__side1 == other.__side2 and self.__side2 == other.__side1 and self.__side3 == other.__side3) or \
               (self.__side1 == other.__side2 and self.__side2 == other.__side3 and self.__side3 == other.__side1) or \
               (self.__side1 == other.__side3 and self.__side2 == other.__side1 and self.__side3 == other.__side2) or \
               (self.__side1 == other.__side3 and self.__side2 == other.__side2 and self.__side3 == other.__side1)

        def __str__(self):
            return "Triangle: side1 = " + str(self.__side1) + " side2 = " + str(self.__side2) + " side3 = " + str(self.__side3)

# Test program

class Triangle:
    def __init__(self, side1, side2, side3, color, filled):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
        self.color = color
        self.filled = filled

    def getArea(self):
        s = (self.side1 + self.side2 + self.side3) / 2
        return (s * (s - self.side1) * (s - self.side2) * (s - self.side3)) ** 0.5

    def getPerimeter(self):
        return self.side1 + self.side2 + self.side3

    def getColor(self):
        return self.color

    def isFilled(self):
        return self.filled

if __name__ == "__main__":
    side1, side2, side3 = eval(input("Enter the three sides of the triangle: "))
    color = input("Enter the color of the triangle: ")
    filled = bool(input("Is the triangle filled (True for yes, False for no)? "))

    triangle = Triangle(side1, side2, side3, color, filled)

    print("Area:", triangle.getArea())
    print("Perimeter:", triangle.getPerimeter())
    print("Color:", triangle.getColor())
    print("Filled:", triangle.isFilled())



########################################################################################
#QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")


import keyword
import re

def count_keywords_identifiers_punctuation(filename):
    keywords = {}
    identifiers = {}
    punctuation = {}

    try:
        f = open(filename, "r")
        for line in f:
            words = line.split()
            for word in words:
                if keyword.iskeyword(word):
                    if word in keywords:
                        keywords[word] += 1
                    else:
                        keywords[word] = 1
                elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", word):
                    if word in identifiers:
                        identifiers[word] += 1
                    else:
                        identifiers[word] = 1
                elif re.match(r"^[^\w\s]$", word):
                    if word in punctuation:
                        punctuation[word] += 1
                    else:
                        punctuation[word] = 1
    except FileNotFoundError:
        print("Invalid filename.")
    finally:
        f.close()

    return keywords, identifiers, punctuation

def main():
    filename = input("Enter the Python source code filename: ")
    keywords, identifiers, punctuation = count_keywords_identifiers_punctuation(filename)
    print("Keywords:", keywords)
    print("Identifiers:", identifiers)
    print("Punctuation:", punctuation)

if __name__ == "__main__":
    main()


########################################################################################
#QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("****************************")


word = input("Enter a string : ")

def is_palindrome(word, counter=0):
    if len(word) <= 1:
        return True
    if word[0] != word[-1]:
        return False
    print("is_palindrome({}, {})".format(word[1:-1], counter+1))
    return is_palindrome(word[1:-1], counter+1)

def main():
    word = "racecar"
    result = is_palindrome(word)
    print("'{}' is a palindrome: {}".format(word, result))

if __name__ == "__main__":
    main()


########################################################################################
#QUESTION V
########################################################################################
print("\n")
print("SOLUTION OF QUESTION V:")
print("****************************")


class NumberOutOfRangeException(Exception):
    def __init__(self, value):
        self.value = value
        self.message = f'{value} is not in the range 2000 < x < 3000'

    def __str__(self):
        return self.message

def read_number():
    while True:
        x = int(input("Enter a number (-9999 to exit): "))
        if x == -9999:
            return
        elif not (2000 < x < 3000):
            raise NumberOutOfRangeException(x)

try:
    read_number()
except NumberOutOfRangeException as e:
    print(e)
