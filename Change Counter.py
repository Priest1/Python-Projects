from tkinter import *

master = Tk()
master.title("Change Counter")
master.configure(width=600,height=400)


def calculatecoins():
    quarter = float(quarters.get())
    cquarter = float(quarter*0.25)
    dime = float(dimes.get())
    cdime = float(dime * 0.1)
    nickel = float(nickels.get())
    cnickel = float(nickel * 0.05)
    penny = float(pennies.get())
    cpenny = float(penny * 0.01)
    total = float(cquarter + cdime + cnickel + cpenny)
    quarterresult=Label(master, text="Quarters Value: $%.2f" % cquarter).grid(row=0, column=4)
    dimeresult=Label(master, text="Dimes Value: $%.2f" % cdime).grid(row=1, column=4)
    nickelresult=Label(master, text="Nickels Value: $%.2f" % cnickel).grid(row=2, column=4)
    pennyresult=Label(master, text="Pennies Value: $%.2f" % cpenny).grid(row=3, column=4)
    totalresult=Label(master, text="Total Value: $%.2f" % total).grid(row=4, column=4)
    return

quarters = StringVar()
dimes = StringVar()
nickels = StringVar()
pennies = StringVar()


lab1 = Label(master, text='Quarters').grid(row=0)
lab2 = Label(master, text='Dimes').grid(row=1)
lab3 = Label(master, text='Nickels').grid(row=2)
lab4 = Label(master, text='Pennies').grid(row=3)


e1 = Entry(master, textvariable=quarters).grid(row=0, column=1)
e2 = Entry(master, textvariable=dimes).grid(row=1, column=1)
e3 = Entry(master, textvariable=nickels).grid(row=2, column=1)
e4 = Entry(master, textvariable=pennies).grid(row=3, column=1)

calcBtn = Button(master, text='Calculate', command=calculatecoins).grid(row=4, column=2)

mainloop()
