def isPalindrome(number):
    return number == number[::-1] # Reverse of a string

while True:
    number = input("Enter A three-digit integer: ")
    if(len(number) == 3):
        break
    
if(isPalindrome(number)):
    print(number, "is a palindrome")
else:
    print(number, "is not a palindrome")