class LinearEquation():
    def __init__(self, a, b, c, d, e, f):
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__e = e
        self.__f = f
    def getA(self):
        return self.__a
    def getB(self):
        return self.__b
    def getC(self):
        return self.__c
    def getD(self):
        return self.__d
    def getE(self):
        return self.__e
    def getF(self):
        return self.__f
    def isSolvable(self):
        return self.__a * self.__d - self.__b * self.__c != 0
    def getX(self):
        return (self.__e * self.__d - self.__b * self.__f) / (self.__a * self.__d - self.__b * self.__c)
    def getY(self):
        return (self.__a * self.__f - self.__e * self.__c) / (self.__a * self.__d - self.__b * self.__c)
    
a,b,c,d,e,f = [float(x) for x in input("Enter a, b, c, d, e, f: ").split(",")]
le = LinearEquation(a,b,c,d,e,f)
if(le.isSolvable()):
    print(f"x is {le.getX()} and y is {le.getY()}")
else:
    print("The equation has no solution")
    

