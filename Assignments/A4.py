########################################################################################
#Name:         Mustaa Furkan BEKER
#Student ID:   61210007
#Department:   Electrical and Electronics Engineering
#Assignment ID: A4
########################################################################################
########################################################################################
#QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")

import tkinter as tk
import tkinter.messagebox as msgbox


def calc_average():
    """
        function to calculate weighted average
        @Param: Marks of different exms and categories
        Return: weighted average with letter grd
    """
    # initialize
    avg = 0

    # if all marks are float/ integer the calculate
    # or throw exception
    try:
        disc = float(discEntry.get())
        revel = float(revelEntry.get())
        quiz = float(quizEntry.get())
        prog = float(projectEntry.get())
        exm1 = float(exm1Entry.get())
        exm2 = float(exm2Entry.get())
        fexm = float(finalEntry.get())

        # calculate
        avg = avg + disc * 0.05
        avg = avg + revel * 0.10
        avg = avg + quiz * 0.10
        avg = avg + prog * 0.15
        avg = avg + exm1 * 0.15
        avg = avg + exm2 * 0.15
        avg = avg + fexm * 0.30

        # find the grd
        grd = str(round(avg, 2)) + "/" + determine_grd(avg)

        # set value
        result.set(grd)

    except ValueError:
        msgbox.showerror("Error!", "Non-Numeric or Blank Data Eneterd")


def determine_grd(average):
    """
        function to calculate letter grd as per average
        @Param: Average weighted score
        Return: Letter grd
    """
    if average >= 89.45:
        return "A"
    elif average >= 79.45:
        return "B"
    elif average >= 69.45:
        return "C"
    elif average >= 59.45:
        return "D"
    else:
        return "F"


def clear_entries():
    """
        function to clear all the field
        @Param:
        Return:
    """
    discEntry.delete(0, "end")
    revelEntry.delete(0, "end")
    quizEntry.delete(0, "end")
    projectEntry.delete(0, "end")
    exm1Entry.delete(0, "end")
    exm2Entry.delete(0, "end")
    finalEntry.delete(0, "end")
    result.set("")


# create a GUI window
root = tk.Tk()
root.title("Weighted Average/grd Calculator")
root.geometry("400x400")

# header of GUI window
intro_label = tk.Label(root, text="Enter your raw score for the following assignments and exms:")
intro_label.grid(columnspan=2, rowspan=2)

# discussion text label and entry
disc_label = tk.Label(root, text="Discussions (5 % of total)")
disc_label.grid(row=2, column=0, columnspan=1)
discEntry = tk.Entry(
    root, textvariable=tk.StringVar())
discEntry.grid(row=2, column=1, columnspan=2)

# revel labs text label and entry
revel_label = tk.Label(root, text="Revel Labs (10 % of total)")
revel_label.grid(row=3, column=0, columnspan=1)
revelEntry = tk.Entry(root, textvariable=tk.StringVar())
revelEntry.grid(row=3, column=1)

# quizzes text label and entry
quiz_label = tk.Label(root, text="     Quizzes (10 % of total)")
quiz_label.grid(row=4, column=0)
quizEntry = tk.Entry(root, textvariable=tk.StringVar())
quizEntry.grid(row=4, column=1)

# project text label and entry
project_label = tk.Label(root, text="     Projects (15 % of total)")
project_label.grid(row=5, column=0)
projectEntry = tk.Entry(root, textvariable=tk.StringVar())
projectEntry.grid(row=5, column=1)

# Exam 1 text label and entry
exm1_label = tk.Label(root, text="     Exam 1 (15 % of total)")
exm1_label.grid(row=6, column=0)
exm1Entry = tk.Entry(root, textvariable=tk.StringVar())
exm1Entry.grid(row=6, column=1)

# Exam 2 text label and entry
exm2_label = tk.Label(root, text="     Exam 2 (15 % of total)")
exm2_label.grid(row=7, column=0)
exm2Entry = tk.Entry(root, textvariable=tk.StringVar())
exm2Entry.grid(row=7, column=1)

# Final Exam text label and entry
final_label = tk.Label(root, text="     Final Exam (30 % of total)")
final_label.grid(row=8, column=0)
finalEntry = tk.Entry(root, textvariable=tk.StringVar())
finalEntry.grid(row=8, column=1)

# result label and entry
result = tk.StringVar()
result_label = tk.Label(root, textvariable=result)
result_label.grid(row=9, column=0, columnspan=2)

# calculate button
calc_button = tk.Button(root, text="Calculate", command=calc_average)
calc_button.grid(row=10, column=0)

# clear button
clear_button = tk.Button(root, text="Clear", command=clear_entries)
clear_button.grid(row=10, column=1)

# quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.grid(row=10, column=2)

root.mainloop()

########################################################################################
#QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")

import math
import random

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)
    return math.sqrt(variance)

def find_min(numbers):
    return min(numbers)

def calculate_correlation(x, y):
    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    diff_x = [(i - mean_x) for i in x]
    diff_y = [(i - mean_y) for i in y]
    product = [(diff_x[i] * diff_y[i]) for i in range(len(x))]
    product_sum = sum(product)
    variance_x = sum(i ** 2 for i in diff_x)
    variance_y = sum(i ** 2 for i in diff_y)
    return product_sum / math.sqrt(variance_x * variance_y)

def main():
    a = [10, 20, 30, 50, 80, 90, 100, 15, 125, 128, 150, 185, 240, 260, 280]
    b = [13, 25, 28, 45, 79, 85, 111, 115, 125, 256, 160, 195, 230, 270, 280, 320]
    c = b.copy()
    c.reverse()
    d = [random.randint(1, 99) for x in range(len(a))]

    print("Lists:")
    print(f"List A = {a}")
    print(f"List B = {b}")
    print(f"List C = {c}")
    print(f"List D = {d}")

    print()
    print(f"List A Average = {calculate_mean(a)}")
    print(f"Standard Deviation of List A = {calculate_standard_deviation(a)}")
    print(f"Minimum of List A = {find_min(a)}")

    print()
    print(f"List B Average = {calculate_mean(b)}")
    print(f"Standard Deviation of List B = {calculate_standard_deviation(b)}")
    print(f"Minimum of List B = {find_min(b)}")

    print()
    print(f"Correlation of List A and B = {calculate_correlation(a, b)}")
    print(f"Correlation of List A and C = {calculate_correlation(a, c)}")
    print(f"Correlation of List A and D = {calculate_correlation(a, d)}")

main()

########################################################################################
#QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")

from collections import defaultdict

import random

# Produce a list of 250 random numbers between 1 and 99.
random_list = [random.randint(1, 100) for _ in range(250)]

# Print out the initial list.
print("Original list:", random_list)

# Make an object called defaultdict that counts how many times each item appears in the list.
counts = defaultdict(int)
for value in random_list:
    counts[value] += 1

# List values that appear more than twice in the code.
duplicates = [key for key, value in counts.items() if value > 2]
print("Duplicate values more than 2 times:", duplicates)

# Eliminate from the list any duplicates that appear more than twice.
filtered_list = [value for value in random_list if counts[value] <= 2]

# Generate the filtered list.
print("List without more than 2 duplicates:", filtered_list)
