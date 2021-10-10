# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:48:24 2021

@author: gcesa
"""
#PROBLEM_1

# Say_Hello_World
print("Hello, World!")

# Python_If_Else
n = int(input().strip())
if n % 2 != 0:
    print("Weird")
else:
    if 2 <= n <= 5:
        print("Not Weird")
    elif 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")

# Arithmetic_Operators
a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)

# Python_Division
a = int(input())
b = int(input())
print(a//b)
print(a/b)

# Loops
n = int(input())
for i in range(n):
    print(i**2)

# Write_a_function


def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0 and year % 400 != 0:
            leap = False
    return leap


year = int(input())
print(is_leap(year))

# Print_function
n = int(input())
string = ''
for i in range(1, n+1):
    string += str(i)
    print(string)

# List_Comprehensions
x = int(input())
y = int(input())
z = int(input())
n = int(input())
perm = []
for x in range(x+1):
    for y in range(y+1):
        for z in range(z+1):
            perm.append([x, y, z])
print([x for x in perm if sum(x) != n])

# Find_the_Runner_Up_Score
n = int(input())
arr = list(map(int, input().split()))
arr = list(set(arr))
arr.sort()
print(arr[-2])


# Nested_Lists
students = []
grades = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    students.append(name)
    grades.append(score)

grades_copy = grades[::]
set_grades = list(set(grades_copy))
set_grades.sort()
second_lowest = set_grades[1]
res = [couple[0] for couple in zip(students, grades) if couple[1] == second_lowest]
res.sort()
for x in res:
    print(x)     

# Finding_the_percentage
n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()
tot = 0
for grade in student_marks[query_name]:
    tot += grade
res = tot / 3
res = format(res, '.2f')
print(res)

# Lists
N = int(input())
res = []
line_check = 0
for line in range(N):
    inst = input().split()
    if inst[0] == 'insert':
        res.insert(int(inst[1]), int(inst[2]))
    elif inst[0] == 'print':
        print(res)
    elif inst[0] == 'remove':
        res.remove(int(inst[1]))
    elif inst[0] == 'append':
        res.append(int(inst[1]))
    elif inst[0] == 'sort':
        res.sort()
    elif inst[0] == 'pop':
        res.pop()
    elif inst[0] == 'reverse':
        res = res[::-1]
    line_check += 1

# Tuples    
n = int(input())
integer_list = map(int, input().split())
t = tuple(integer_list)
print(hash(t))

# sWAP_cASE
def swap_case(s):
    new_s = ''
    for letter in s:
        if letter.isupper():
            new_s += letter.lower()
        else:
            new_s += letter.upper()
    return new_s


if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String_Split_and_Join
def split_and_join(line):
    splitted = line.split(" ")
    return '-'.join(splitted)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's_Your_Name
def print_full_name(first, last):
    name = first + ' ' + last
    print('Hello ' + name + '! You just delved into python.')

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations
def mutate_string(string, position, character):
    new_string = string[:position] + character + string[position+1:]
    return new_string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find_a_String
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)):
        if string[i:i+len(sub_string)] == sub_string:
            count +=1
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
    
# String_Validators
    s = input()
    check_alnum = [l.isalnum() for l in s]
    if sum(check_alnum) > 0: print(True)
    else: print(False)
    check_alpha = [l.isalpha() for l in s]
    if sum(check_alpha) > 0: print(True)
    else: print(False)
    check_dig = [l.isdigit() for l in s]
    if sum(check_dig) > 0: print(True)
    else: print(False)
    check_lower = [l.islower() for l in s]
    if sum(check_lower) > 0: print(True)
    else: print(False)
    check_upper = [l.isupper() for l in s]
    if sum(check_upper) > 0: print(True)
    else: print(False)

# Text_Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# Text_Wrap
import textwrap

def wrap(string, max_width):
    res = ''
    for i in range(0, len(string), max_width):
        res += string[i:i+max_width] + '\n'
    return res[:-1]

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# String_Formatting
def print_formatted(number):
    res = ''
    width = len(bin(number)[2:])
    for i in range(1,number+1):
        res += str(i).rjust(width, ' ') + ' '
        res += oct(i)[2:].rjust(width, ' ') + ' '
        res += hex(i).upper()[2:].rjust(width, ' ') + ' '
        res += bin(i)[2:].rjust(width, ' ') + '\n'
    print(res)

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# Capitalize!
import math
import os
import random
import re
import sys

def solve(s):
    res = []
    for w in s.split(" "):
        if w != "":
           res.append(w[0].upper()+w[1:])
        else:
            res.append(w)
    return ' '.join(res)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


# Merge_the_Tools
def merge_the_tools(string, k):
    substrings = []
    for i in range(0, len(string), k):
        substrings.append(string[i:i+k])
    for st in substrings:
        s = []
        for l in st:
            if l not in s:
                s.append(l)
        print(''.join(s))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
        
# Minion_Game
vowels = 'AEIOU'

def minion_game(string):
    s = 0
    k = 0
    for i in range(len(string)):
        if string[i] in vowels:
            k += len(string[i:])
        else:
            s += len(string[i:])
    if s > k: print('Stuart' + ' ' + str(s))
    elif s == k: print('Draw')
    else: print('Kevin' + ' ' + str(k))

if __name__ == '__main__':
    s = input()
    minion_game(s)

# Introduction_to_Sets
def average(array):
    ins = set(array)
    return sum(ins)/len(ins)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# No_idea
n, m = list(map(int, input().split()))
arr = list(map(int, input().split()))
set_a = set(map(int, input().split()))
set_b = set(map(int, input().split()))
happiness = 0
for i in range(len(arr)):
    if arr[i] in set_a:
        happiness += 1
    elif arr[i] in set_b:
        happiness -= 1

print(happiness)

# Symmetric_Difference
M, set_M = int(input()), set(list(map(int, input().split())))
N, set_N = int(input()), set(list(map(int, input().split())))
diff = set_M.difference(set_N).union(set_N.difference(set_M))
for el in sorted(diff):
    print(str(el))
    
# Set.add()
N = int(input())
res = set()
for i in range(N):
    res.add(input())
print(len(res))

# Set.discard()_.remove()_and_.pop()
n = int(input())
s = set(map(int, input().split()))
ins = set(s)
n_cmd = int(input())
for i in range(n_cmd):
    inst = input().split()
    cmd = inst[0]
    #print(cmd)
    if cmd == 'pop':
        ins.pop()
    elif cmd == 'remove':
        el = inst[1]
        ins.remove(int(el))
    elif cmd == 'discard':
        el = inst[1]
        ins.discard(int(el))
    
print(sum(ins))

# Set.union()_Operation
n_e, set_e = int(input()), set(input().split())
n_f, set_f = int(input()), set(input().split())
print(len(set_e.union(set_f)))

# Set.intersection()_Operation
n_e, set_e = int(input()), set(input().split())
n_f, set_f = int(input()), set(input().split())
print(len(set_e.intersection(set_f)))

# Set.difference()_Operation
n_e, set_e = int(input()), set(input().split())
n_f, set_f = int(input()), set(input().split())
print(len(set_e.difference(set_f)))

# Set.symmetric_difference()_Operation
n_e, set_e = int(input()), set(input().split())
n_f, set_f = int(input()), set(input().split())
print(len(set_e.symmetric_difference(set_f)))

# Set_Mutations
n_s, first_set = int(input()), set(map(int, input().split()))
tot_n = int(input())


for i in range(tot_n):
    inst = input().split()
    other_set = set(map(int, input().split()))
    if inst[0] == 'update':
        first_set.update(other_set)
    elif inst[0] == 'intersection_update':
        first_set.intersection_update(other_set)
    elif inst[0] == 'symmetric_difference_update':
        first_set.symmetric_difference_update(other_set)
    elif inst[0] == 'difference_update':
        first_set.difference_update(other_set)
        
        
print(sum(first_set))

# The_Captain's_Room
K = int(input())
room_list = input().split()

from collections import Counter

count = Counter(room_list)
res = [x for x in list(count.keys()) if count[x] == 1]
print(res[0])

# Check_subset
n_t = int(input())
for i in range(n_t):
    n_a, set_a = int(input()), set(map(int, input().split()))
    n_b, set_b = int(input()), set(map(int, input().split()))
    print(set_a.intersection(set_b) == set_a)
    
# Check_strict_Superset
set_a = set(map(int, input().split()))
n = int(input())
res = []
for i in range(n):
    set_b = set(map(int, input().split()))
    if set_a.intersection(set_b) == set_b and set_a.difference(set_b) != set():
        res.append(True)
    else:
        res.append(False)

if all(res): print(True)
else: print(False)

# Collections.Counter()
from collections import Counter
x = int(input())
sizes = list(map(int, input().split()))
n = int(input())
shoe_d = Counter(sizes)
res = 0
for i in range(n):
    desired, price = list(map(int, input().split()))
    if desired in list(shoe_d.keys()) and shoe_d[desired] > 0:
        res += price
        shoe_d[desired] -= 1
print(res)

# DefaultDict_Tutorial
from collections import defaultdict
n, m = list(map(int, input().split()))
group_a = [input() for i in range(n)]
group_b = [input() for i in range(m)]
diz = defaultdict(list)
for i in range(len(group_a)):
    key = group_a[i]
    diz[key].append(str(i+1))
for letter in group_b:
    if letter in list(diz.keys()):
        print(' '.join(diz[letter]))
    else:
        print('-1')
        
# Collections.namedtuple()
from collections import namedtuple

n = int(input())
student = namedtuple('student', input().split())
tot = 0
for i in range(n):
    s = student(*input().split())
    tot += int(s.MARKS)
print(round(tot/n,2))    

# Collections.OrderedDict()
from collections import OrderedDict

n = int(input())
items_d = OrderedDict()
for i in range(n):
    cmd = input().split()
    price = int(cmd[-1])
    item = ' '.join(cmd[:-1])
    if item in list(items_d.keys()):
        items_d[item] += price
    else:
        items_d[item] = price
for k, el in list(items_d.items()):
    print(k, el)

# Word_Order
from collections import Counter, OrderedDict
n = int(input())
words = [input() for i in range(n)]
word_d = Counter(words)
ordered = OrderedDict.fromkeys(words)
res = [str(word_d[key]) for key in list(ordered.keys())]
print(len(res))
print(' '.join(res))

# Collections.deque()
from collections import deque

n = int(input())
deq = deque()
for i in range(n):
    cmd = input().split()
    if cmd[0] == 'append':
        deq.append(cmd[1])
    elif cmd[0] == 'pop':
        deq.pop()
    elif cmd[0] == 'popleft':
        deq.popleft()
    elif cmd[0] == 'appendleft':
        deq.appendleft(cmd[1])
print(' '.join(deq))

# Company_Logo
import math
import os
import random
import re
import sys

from collections import Counter, OrderedDict


if __name__ == '__main__':
    s = input()
    letters = [l for l in s]
    diz = Counter(letters)
    ordered = OrderedDict({key : el for key, el in sorted(diz.items(), key = lambda x : (-x[1], x))})
    for i in range(3):
        print(list(ordered.keys())[i] + ' ' + str(list(ordered.values())[i]))

# Calendar_Module
import calendar
M, D, Y = list(map(int, input().split()))
res = str((calendar.weekday(Y, M, D)))
d_tran = {'0' : 'MONDAY', '1' : 'TUESDAY', '2' : 'WEDNESDAY', '3' : 'THURSDAY', '4' : 'FRIDAY', '5' : 'SATURDAY', '6' : 'SUNDAY'}
tran = str.maketrans(d_tran)
print(res.translate(tran))

# Exceptions
n = int(input())
for i in range(n):
    try:
        a,b = list(map(int, input().split()))
        res = a//b
        print(res)
    except ValueError as e:
        print("Error Code:", e)
    except ZeroDivisionError as e:
        print("Error Code:", e)
        
# Zipped!
n, x = list(map(int, input().split()))
subjects = []
for i in range(x):
    grades = list(map(float, input().split()))
    subjects.append(grades)
for student in zip(*subjects):
    print(sum(student)/x)
    
# ginortS
s = input()
s = sorted(s, key = lambda x : (x.isnumeric(), x.isupper(), x))
letters = [l for l in s if l.isalpha()]
digits = [d for d in s if d.isnumeric()]
digits = sorted(digits, key = lambda x : (-(int(x) % 2), x))
res = ''.join(letters) + ''.join(digits)
print(res)

# Map_and_Lambda_Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    if n == 0:
        return []
    if n == 1:
        return [0]
    else:
        seq = [0,1]
        for i in range(n-2):
           seq.append(seq[-1]+seq[-2])
    return seq

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Re.split()
# Re was an unknown package to me, so I actually used the 'Discussions' tab on more complicated exercises
# to get hints on how to write the regular expressions and actually familiarize a little bit with the syntax.

regex_pattern = r"[,.]"	# Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))

# Re.start()_and_Re.end()
import re
s = input()
k = input()
pattern = '('+'?='+'('+k+')'+')'
it = [x for x in re.finditer(pattern,s)]
if it:
    for x in it:
        print((x.start(), x.end()+len(k)-1))
else:
     print((-1,-1))       

# Group()_Groups()_and_Groupdict()
import re
s=input()
m=re.match(r'.*?([a-z0-9A-Z])\1', s)
if m: print(m.group(1))
else: print(-1)

# Re.findall()_and_Re.finditer()
import re
pattern = re.compile('[aeiouAEIOU]{2,}[^aeiouAEIOU]{1}')
it = [x for x in re.finditer(pattern, input())]
if it:
    for m in it:
        print(m.group()[:-1])
else:
    print(-1)

# Detect_Floating_Point_Number
import re
t = int(input())
p = re.compile(r'^[-+\.]?[0-9]*\.[0-9]+$')
for i in range(t):
   m = p.match(input())
   if m: print(True)
   else: print(False)
   
# Validating_Roman_Numerals
regex_pattern = r"M{,3}(CM|CD|D?C{,3})(XC|XL|L?X{,3})(IX|IV|V?I{,3})$"
import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating_Phone_Numbers
import re
n = int(input())
pattern = r'[789]\d{9}$'
for i in range(n):
    num = input()
    m = re.match(pattern,num)
    if m:
        print('YES')
    else:
        print('NO')
        
# Validating_and_Parsing_Email_Addresses
import re
n = int(input())
pattern = r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>'
for i in range(n):
    line = input().split()
    name = line[0]
    add = line[1]
    m = re.match(pattern, add)
    if m:
        print(name + ' ' + add)
    else:
        pass
    
# Hex_Color_Code
import re
n = int(input())
for i in range(n):
    code = input()
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', code)
    if matches:
        for x in matches:
            print(x)
            
# Validating UID
import re
n = int(input())
pattern = [r'^[a-zA-Z0-9]{10}$', r'(.*[A-Z].*){2,}',r'(.*[0-9].*){3,}', r'^(?:([a-zA-Z0-9])(?!.*\1))*$']
for i in range(n):
    test = input()
    match = [re.match(p,test) for p in pattern]
    if all(match):
        print('Valid')
    else:
        print('Invalid')
        
# Regex_Substitution
import re
n = int(input())
text = ''
for i in range(n):
    text += input() + '\n'
text = re.sub(r'(?<= )(&&)(?= )', 'and', text)
text = re.sub(r'(?<= )(\|\|)(?= )', 'or', text)
print(text)
        
#XML_1_Find_The_Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    count = 0
    for el in node.iter():
        count += len(el.items())
    return count

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# Arrays
import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr[::-1], float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape_and_Reshape
import numpy

arr = numpy.array(list(map(int,input().split())))
print(arr.reshape(3,3))


# Transpose_and_Flatten
import numpy


n, m = list(map(int, input().split()))
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
    
arr = numpy.array(l)
print(numpy.transpose(arr))
print(arr.flatten())

# Concatenate
import numpy

n, m, p = list(map(int, input().split()))
l1 = [list(map(int, input().split())) for i in range(n)]
l2 = [list(map(int, input().split())) for i in range(m)]
arr1 = numpy.array(l1)
arr2 = numpy.array(l2)
print(numpy.concatenate((arr1, arr2), axis = 0))

# Zeros_and_Ones
import numpy

dimensions = list(map(int, input().split()))
print(numpy.zeros((dimensions), dtype = int))
print(numpy.ones((dimensions), dtype = int))

# Eye_and_Identity
import numpy
numpy.set_printoptions(legacy='1.13')

n, m = list(map(int, input().split()))
if n == m:
    print(numpy.identity(n))
else:
    print(numpy.eye(n,m))

# Array_Mathematics
import numpy
n, m = list(map(int, input().split()))
l1 = []
l2 = []
for i in range(n):
    l1.append(list(map(int, input().split())))
for i in range(n):
    l2.append(list(map(int, input().split())))
arr1 = numpy.array(l1)
arr2 = numpy.array(l2)

print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1//arr2)
print(arr1 % arr2)
print(arr1 ** arr2)

# Floor,Ceil_and_Rint
import numpy
numpy.set_printoptions(legacy='1.13')

arr = numpy.array(list(map(float, input().split())))
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))


# Sum_and_Prod
import numpy

n,m = list(map(int, input().split()))
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
s = numpy.sum(arr, axis = 0)
print(numpy.prod(s))


# Min_and_Max
import numpy

n, m = list(map(int, input().split()))
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
mini = numpy.min(arr, axis = 1)
print(numpy.max(mini))

# Mean_Var_and_Std
import numpy

n,m = list(map(int, input().split()))
l = []
for i in range(n):
    l.append(list(map(int, input().split())))
arr = numpy.array(l)
print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(round(numpy.std(arr), 11))

# Dot_and_Cross
import numpy
n = int(input())
l_a = [list(map(int, input().split())) for i in range(n)]
l_b = [list(map(int, input().split())) for i in range(n)]
a = numpy.array(l_a)
b = numpy.array(l_b)
print(numpy.dot(a,b))

# Inner_and_Outer
import numpy
a = numpy.array(list(map(int, input().split())))
b = numpy.array(list(map(int, input().split())))
print(numpy.inner(a,b))
print(numpy.outer(a,b))

# Polynomials
import numpy
n = list(map(float, input().split()))
x = float(input())
print(numpy.polyval(n,x))


# Linear_Algebra
import numpy
n = int(input())
l = []
for i in range(n):
    l.append(list(map(float, input().split())))
    
arr = numpy.array(l)
print(round(numpy.linalg.det(arr),2))




#PROBLEM_2

# Birthday_Cake_Candles
import math
import os
import random
import re
import sys

from collections import Counter

def birthdayCakeCandles(candles):
    diz = Counter(candles)
    ordered_keys = sorted(list(diz.keys()), reverse=True)
    return int(diz[ordered_keys[0]])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number_Line_Jumps
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    for i in range(10000):
        x1 += v1
        x2 += v2
        if x1 == x2:
            return 'YES'
            break
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
    
# Viral_Advertising
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    shared = 5
    liked = math.floor(5/2)
    tot = liked
    for i in range(2,n+1):
        shared = liked * 3
        liked = math.floor(shared/2)
        tot += liked
    return tot

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
# Recursive_Digit_Sum
import math
import os
import random
import re
import sys

def superDigit(n, k):
    p = n * k
    return digitsum(p)
    
def digitsum(n):
    if len(str(n)) == 1:
        return n
    else:
        first = int(str(n)[0])
        rest = int(str(n)[1:])
        first += digitsum(rest)
        return digitsum(first)
    
            
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
    
# Insertion_Sort_Part_1
import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    e = arr[-1]
    for i in range(n-1,-1,-1):
        if arr[i-1] < e:
            arr[i] = e
            print(' '.join(map(str, arr)))
            break
        if i == 0:
            arr[i] = e
            print(' '.join(map(str, arr)))
        else:
            arr[i] = arr[i-1]
            print(' '.join(map(str,arr)))
        

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
    
# Insertion_Sort_Part_2
import math
import os
import random
import re
import sys


def insertionSort2(n, arr):
    for i in range(1,n):
        val = arr[i]
        for j in range(i):
            if arr[j] > val:
                arr.remove(arr[i])
                arr.insert(j, val)
                break
        print(' '.join(map(str,arr)))
                
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)













