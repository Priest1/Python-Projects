while True:
    uin = input(">> ") #user input
    try:
            when_to_stop = abs(int(uin))
    except KeyboardInterrupt:
            break
    except:
            print("Not a number")
    while when_to_stop > 0:
       m, s = divmod(when_to_stop, 60)
       h, m = divmod(m, 60)
        print(str(h) + ":" + str(m) + : + str(s))
        when_to_stop -= 1