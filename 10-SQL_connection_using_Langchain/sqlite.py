import sqlite3

connection = sqlite3.connect('student.db')

cursor = connection.cursor()

table_creation = """
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_creation)

cursor.execute("INSERT INTO STUDENT VALUES('Aman', 'Machine Learning', 'B', 87)")
cursor.execute("INSERT INTO STUDENT VALUES('Riya', 'Data Science', 'A', 94)")
cursor.execute("INSERT INTO STUDENT VALUES('Vikram', 'Cyber Security', 'C', 72)")
cursor.execute("INSERT INTO STUDENT VALUES('Neha', 'Web Development', 'A', 91)")
cursor.execute("INSERT INTO STUDENT VALUES('Manish', 'Blockchain', 'B', 83)")
cursor.execute("INSERT INTO STUDENT VALUES('Sneha', 'AI', 'A', 95)")
cursor.execute("INSERT INTO STUDENT VALUES('Arjun', 'Data Analytics', 'D', 67)")
cursor.execute("INSERT INTO STUDENT VALUES('Pooja', 'IoT', 'B', 79)")
cursor.execute("INSERT INTO STUDENT VALUES('Ravi', 'Cloud Computing', 'A', 89)")
cursor.execute("INSERT INTO STUDENT VALUES('Divya', 'DevOps', 'C', 74)")


print('The inserted records are')

data = cursor.execute('''Select * from STUDENT''')

for row in data:
    print(row)

connection.commit()
connection.close()