def LetterGradeToScore(letter):
    if letter == "A": return 4.00
    if letter == "B": return 3.50
    if letter == "C": return 3.00
    if letter == "D": return 2.50
    if letter == "F": return 0.00
    
numOfStudents = int(input("Number Of Students: "))
overall = []

for i in range(numOfStudents):
    score = 0.00
    for j in range(3):
        course = input("Course: ")
        grade = input("Grade: ")
        score = score + float(LetterGradeToScore(grade))
    overall.append(score/3)
    
for i in range(numOfStudents):
    print("Calculated Overall Score for student",i+1,":",round(overall[i], 2))
