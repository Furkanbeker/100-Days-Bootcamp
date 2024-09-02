#QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")

from random import randint
def printMatrix(n):
    for i in range(n):
        for j in range(n):
            print(randint(0,1),end=' ')
        print(' ')
n=int(input('Enter n: '))
printMatrix(n)

########################################################################################
#QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")


def binaryToHexDec(binaryvalue):
    for i in range(len(binaryValue)):
        if not (binaryValue[i] == "1" or binaryValue[i] == "0"):
            print("\nInvalid Input!")
            return 0, 0, 0

    hexval = hex(int(binaryValue, 2))

    decVal = int(binaryValue, 2)

    octal = oct(int(binaryValue, 2))

    return hexVal, decVal, octal


binaryValue = input("Enter a binary number: ")

hexVal, decVal, octal = binaryToHexDec(binaryValue)

print("\nThe hexadecimal equivalent is =", hexVal)
print("The decimal equivalent is =", decVal)
print("The octal equivalent is =", octal)


########################################################################################
#QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")


def common_prefix(s1, s2):
    res = ""
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            res = res + s1[i]
        else:
            return res
    return res


if __name__ == "__main__":
    s1 = input("Enter string1: ")
    s2 = input("Enter string2: ")
    print(common_prefix(s1, s2))

########################################################################################
#QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("****************************")

# In[ ]:

class Customer:
    """Customer class"""
    def __init__(self, id: int, name: str, discount: int):
        """constructor"""
        self.__id = id
        self.__name = name
        self.__discount = discount

    def getID(self) -> int:
        """give the customer's ID back"""
        return self.__id

    def getName(self) -> str:
        """return the customer's name"""
        return self.__name

    def getDiscount(self) -> int:
        """return the customer's discount"""
        return self.__discount

    def setDiscount(self, discount: int) -> None:
        """set the discount """
        self.__discount = discount

    def toString(self) -> str:
        """return Customer information as string"""
        return f'{self.__name}({self.__id})({self.__discount}%)'


class Invoice:
    """Invoice class"""
    def __init__(self, id: int, customer: Customer, amount: float):
        """constructor"""
        self.__id = id
        self.__customer = customer
        self.__amount = amount

    def getID(self) -> int:
        """return the ID of the Invoice"""
        return self.__id

    def getCustomer(self) -> Customer:
        """return the customer"""
        return self.__customer

    def setCustomer(self, customer: Customer) -> None:
        """set the customer """
        self.__customer = customer

    def getAmount(self) -> float:
        """return the amount"""
        return self.__amount

    def setAmount(self, amount: float):
        """set the amount """
        self.__amount = amount

    def getCustomerID(self) -> int:
        """give the customer's ID back"""
        return self.__customer.getID()

    def getCustomerName(self) -> str:
        """return the name of the customer"""
        return self.__customer.getName()

    def getCustomerDiscount(self) -> int:
        """return the customer's discount"""
        return self.__customer.getDiscount()

    def getAmountAfterDiscount(self) -> float:
        """return the amount after discount"""
        return self.__amount * (1 - self.__customer.getDiscount() * 0.01)

    def toString(self) -> str:
        """return the invoice details as a string"""
        return f'Invoice[id={self.__id}, customer = {self.__customer.toString()}, amount={self.__amount}]'


""" test code """
# Create two instances of the Customer.
cust1 = Customer(123, "John Doe", 10)
cust2 = Customer(435, "Jerry Smith", 15)
# Print customer ID
print("Customer ID = ", cust1.getID())
print("Customer Name = ", cust1.getName())
print("Customer discount = ", cust1.getDiscount())
# Produce three instances of Invoice.
inv1 = Invoice(6011, cust1, 500)
inv2 = Invoice(6022, cust2, 850)
inv3 = Invoice(6033, cust1, 350)
#  Print the details of the customer.
print("Customer ID = ", inv2.getCustomerID())
print("Customer Name = ", inv2.getCustomerName())
print("Customer discount = ", inv2.getCustomerDiscount())
# Print the invoices with the discounted amount.
print(inv1.toString())
print("After discount amount = $%.2f" % inv1.getAmountAfterDiscount())
print(inv2.toString())
print("After discount amount = $%.2f" % inv2.getAmountAfterDiscount())
print(inv3.toString())
print("After discount amount = $%.2f" % inv3.getAmountAfterDiscount())
# Change the data.
cust1.setDiscount(20)
inv3.setCustomer(cust1)
inv3.setAmount(1250)
# Print the invoice with the discounted amount.
print(inv3.toString())
print("After discount amount = $%.2f" % inv3.getAmountAfterDiscount())



