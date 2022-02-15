# The purpose of this program is to ask the user for a series of inputs and generate a password based on those inputs
from random import randint

password = ""
# Ask how long password needs to be.
length = input("How long does the password need to be? ")
numbers = input("How many numbers does it need? ")
special = input("How many special characters does it need? ")
for h in range(int(numbers)):
    h = str(randint(0,9))
    for g in range(int(special)):
        g = chr(randint(32,38))
        for i in range(int(length)):
            i = chr(randint(65, 90))
    password = str(password) + h + g + i - len(h) - len(g)
print(password())


