class Animal:
    def __init__(self):
        self.num_eyes = 2

    def breathe(self):
        print("a b c")

class Fish(Animal):
    def __init__(self):
        super().__init__()
        
    def breathe(self):
        super().breathe()
        print("under water")

    def swim(self):
        print("x y z ")

pink = Animal()
pink.breathe()

nemo = Fish()
nemo.swim()
nemo.breathe()

print(nemo.num_eyes)