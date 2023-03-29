import RPi.GPIO as GPIO
import time
from tkinter import *
led = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led, GPIO.OUT)
GPIO.setwarnings(False)

def off():
    GPIO.output(led, not(GPIO.input(led)))
    print (GPIO.input(led))

root = Tk()

btnOff = Button(root, text="Off Button", command=off)
btnOff.pack()
root.mainloop()







