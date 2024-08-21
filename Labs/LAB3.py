def celsiusToFahrenheit(celsius):
    fahrenheit = (9 / 5) * celsius + 32
    return celsius, fahrenheit

def FahrenheitToCelsius(fahrenheit):
    celsius = (5 / 9) * (fahrenheit - 32)
    return fahrenheit, celsius

fill = ' '
align = '<'
width = 15
j = 120
print(f'{"Celsius":{fill}{align}{width-2}}', f'{"Fahrenheit":{fill}{align}{width}}', f'{"|":{fill}{align}{5}}' , end="")
print(f'{"Fahrenheit":{fill}{align}{width}}',f'{"Celsius":{fill}{align}{width}}')
for i in range(40, 30, -1):  
    print(f'{celsiusToFahrenheit(i)[0]:{fill}{align}{width}}', end = "")    
    print(f'{"{:.2f}".format(celsiusToFahrenheit(i)[1]):{fill}{align}{width}}', end = "")
    print(f'{"|":{fill}{align}{5}}', end = "")
    print(f'{FahrenheitToCelsius(j)[0]:{fill}{align}{width}}', end = "")    
    print(f'{"{:.2f}".format(FahrenheitToCelsius(j)[1]):{fill}{align}{width}}',)    
    j = j-10

