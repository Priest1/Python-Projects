def main():
    fahrenheit = float(input("Enter Fahrenheit temperature "))     # Prompt for and Fahrenheit temperature
    celsius = ((fahrenheit - 32) * 5) / 9                          # Converts from Fahrenheit to Celsius
    result = round(celsius, 1)                                     # Rounds to one decimal place
    
   
    print(result)
    main()

main()

