# Disassembly module

python -m dis 'path_to\\file.py'


# Lib version

print(numpy.__version__)



# Documentation

def f():
    ''' Some docs '''
    return 1909
    
print(f.__doc__)  # also for classes and its methods



# Упаковка и распаковка аргументов

def f(*args):  # type(args) == tuple
    for arg in args:
        # do smth

f(1, 2, 'd')

x = [1, 2, 'd']  # any iterator
f(*x)



# Области видимости - 4: Local, Enclosing, Global, Builtin

min = 1
def f():
    min += 1   # min = min + 1 = 2, но это уже локальная переменная; глобальная останется 1
    
надо:

def f():
    global min
    min += 1
    return min


# Уникальный идентификатор (=адрес) объекта

id(x)
    

# Расширенная упаковка/распаковка

first, *rest, last = range(1, 10)


# Одномерный кортеж 
x = (10,)  # без запятой было бы просто int
y = x + (2, 'f', True)


# Apply function to list

xs = [1, 2, 3, 4, 5]
y_map = map(lambda x: x**2, xs)


# Delete item from list by index

del x[5]  # x.pop(5)


# Dict methods

x.keys()
x.values()
x.items()


# Reverse collection/strings

reversed(x)


# Try get value from dict with default

x.get('my_key', 123)


# Filter list

nums = [1, 2, 3, 4, 5]
x = list(filter(lambda x: x>3, nums))  # [4 5]
y = list(filter(None, [0, None, [], {}, set(), '', 42]))  # [42] - filter out falsy values


# Zip

x = [1, 3, 5]
y = ['a', 'c', 'z']
z = list(zip(x, y))   # [(1, 'a'), (3, 'c'), (5, 'z')]


# Unzip

x = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
zip(*x)   # [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]


# Looping

[ x**2 for x in range(1, 9)]
[ x**2 for x in range(1, 9) if x%2==0]
{ x**2 for x in range(1, 9) }
{ x: x**2 for x in range(1, 9)}

# Create dictionary from iterable

d = dict((x, some_mapping(x)) for x in xs)


# Sort dict by value

for key, value in sorted(my_dict.items(), key=lambda kv: kv[1], reverse=True):   # descending
    # do smth.


# Read file contents

with file = open('path/to/file.txt') as file:
    line1 = file.readline() #read 1st line 
    contents = file.read()  #read the least of the file to the end


# Iterate through file lines

with open('path/to/file.txt') as file:
    for line in file:
        print(line.strip()) # lines without linebreaks


# Read file in UTF8

import codecs

with file = open('path/to/file.txt', 'r', encoding='utf8') as file:
    contents = file.read()


# Write to file

with open('path/to/file.txt', 'w') as file:   
    file.write('string to write')
    
with open('path/to/file.txt', 'a') as file:   
    file.write('another string to append')

    
# Временный файл (удаляется после выходя из with)

import tempfile

with tempfile.TemporaryFile() as handle:
   # do smth
    

# Value counts in a list

from collections import Counter

diff_names = Counter(names)


# Enumerate some collection along with indices

xs = [1, 2, 3, 6, 8, 0]
for i, x in enumerate(xs):
    # do smth


# Singleton (using metaclass)

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Utils(metaclass=Singleton):
    def do_work(self):
        pass

        
# Singleton (using decorator)

def singleton(cls):
    instance = None

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance
        
    return inner
    
@singleton
class A:
    # some
    
    
id(A())  # 55545654756
id(A())  # 55545654756


# Itertools
import itertools as it

islice - слайс коллекции (как [::])
take   - взять сколько-то штук из коллекции
drop   - пропустить сколько-то штук из коллекции
chain  - объединение коллекций в линию
product([1, 2], repeat = 3)   # [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), ...]  - x8
combinations([1, 2, 3], 2)   # [(1, 2), (1, 3), (2, 3)]


