from tkinter import *
from tkinter import messagebox
from random import choice, randint, shuffle
import pyperclip

def gnrt_password():
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

    password_letters = [choice(letters) for a in range(randint(2, 5))]
    password_symbols= [choice(symbols) for b in range(randint(2, 4))]
    password_numbers= [choice(numbers) for c in range(randint(2, 4))]

    password_list = password_symbols + password_numbers + password_letters
    shuffle(password_list)

    psword = "".join(password_list)
    password_entry.insert(0, psword)
    pyperclip.copy(password)

window = Tk()
window.title("Password Manager")
window.config(padx=50, pady=50, bg= "white")

canvas = Canvas(width=200 , height=224, bg="white", highlightthickness=0)
imagename = PhotoImage(file="logo.png")
canvas.create_image(100, 100, image=imagename)
canvas.grid(column=1, row = 0)

website = Label(text="Website:", bg= "white" )
website.grid(column = 0, row=1 )
website_entry = Entry(width=40, bg="white")
website_entry.grid(column=1 , row=1, columnspan=2)

mail = Label(text= "Email/Username:", bg="white")
mail.grid(column= 0, row =2)
mail_entry = Entry(width=40 , bg="white")
mail_entry.grid(column=1, row=2, columnspan=2)
mail_entry.insert(0, "try@gmail.com")

password=Label(text= "Password:", bg="white")
password.grid(column=0 , row = 3)
password_entry = Entry(width=21 , bg = "white")
password_entry.grid(column=1 , row=3)

generate_password=Button(text="Generate Password", bg="white", width=15, command=gnrt_password)
generate_password.grid(column = 2, row=3,)


def save():

    website = website_entry.get()
    mail = mail_entry.get()
    password = password_entry.get()

    if mail=="" or password=="" or website=="":
        messagebox.showinfo(title=website, message="Please don't leave any fields empty!")

    else:
        is_ok = messagebox.askokcancel(title=website, message=f"These are the details entered: \nEmail:{mail}"
                                                  f"\nPassword: {password} \nIs it ok to save?")

        if is_ok:
            with open("passwords.txt", "a") as file:
                file.write(f"{website} \ {mail} \ {password}\n")
                website_entry.delete(0, END)
                password_entry.delete(0, END)


add_password= Button(text="Add", bg="white", width=35, command=save)
add_password.grid(column=1, row=4, columnspan=2)


window.mainloop()
