########################################################################################
# Name:         Mustafa Furkan BEKER
# Student ID:   61210007
# Department:   Electrical and Electronics Engineering
# Assignment ID: A1
########################################################################################
########################################################################################
# QUESTION I
########################################################################################
print("\n")
print("SOLUTION OF QUESTION I:")
print("****************************")


import random
import time

def fillFile(fileSize, fileName):
    nums = [str(random.randint(0, fileSize + 1000)) + '\n' for _ in range(fileSize)]
    with open(fileName, 'w') as f:
        f.writelines(nums)

def readFile(fileName):
    nums = []
    try:
        f = open(fileName, 'r')
        for line in f:
            nums.append(int(line.strip()))
        f.close()
    except IOError:
        print("Error: File not found")
    return nums


fileSizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
statsFile = open('fileStats.txt', 'w')

for size in fileSizes:
    start = time.time()
    fillFile(size, f'file{size}')
    end = time.time()
    fillTime = end - start
    statsFile.write(f'fillFile {fillTime:.5f} ')
    start = time.time()
    nums = readFile(f'file{size}')
    end = time.time()
    readTime = end - start
    statsFile.write(f'readFile {readTime:.5f}\n')
    assert len(nums) == size

statsFile.close()

########################################################################################
#QUESTION II
########################################################################################
print("\n")
print("SOLUTION OF QUESTION II:")
print("****************************")

import math

def bubble_sort(array):
    swapped = False
    for i in range(len(array) - 1, 0, -1):
        for j in range(i):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
        if swapped:
            swapped = False
        else:
            break
    return array


def selection_sort(array):
    for i in range(len(array)):
        min_index = i
        for j in range(i + 1, len(array)):
            if array[j] < array[min_index]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i]
    return array


def insertion_sort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while array[j] > key and j >= 0:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
    return array


def shell_sort(array):
    n = len(array)
    k = int(math.log2(n))
    interval = 2 ** k - 1
    while interval > 0:
        for i in range(interval, n):
            temp = array[i]
            j = i
            while j >= interval and array[j - interval] > temp:
                array[j] = array[j - interval]
                j -= interval
            array[j] = temp
        k -= 1
        interval = 2 ** k - 1
    return array


def mergeSort(nums):
    if len(nums) == 1:
        return nums
    mid = (len(nums) - 1) // 2
    lst1 = mergeSort(nums[:mid + 1])
    lst2 = mergeSort(nums[mid + 1:])
    result = merge(lst1, lst2)
    return result


def merge(lst1, lst2):
    lst = []
    i = 0
    j = 0
    while (i <= len(lst1) - 1 and j <= len(lst2) - 1):
        if lst1[i] < lst2[j]:
            lst.append(lst1[i])
            i += 1
        else:
            lst.append(lst2[j])
            j += 1
    if i > len(lst1) - 1:
        while (j <= len(lst2) - 1):
            lst.append(lst2[j])
            j += 1
    else:
        while (i <= len(lst1) - 1):
            lst.append(lst1[i])
            i += 1
    return lst


def quickSort(array):
    if len(array) > 1:
        pivot = array.pop()
        grtr_lst, equal_lst, smlr_lst = [], [pivot], []
        for item in array:
            if item == pivot:
                equal_lst.append(item)
            elif item > pivot:
                grtr_lst.append(item)
            else:
                smlr_lst.append(item)
        return (quickSort(smlr_lst) + equal_lst + quickSort(grtr_lst))
    else:
        return array


def heapify(array, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and array[i] < array[l]:
        largest = l
    if r < n and array[largest] < array[r]:
        largest = r

    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        heapify(array, n, largest)


def heap_sort(array):
    n = len(array)
    for i in range(n // 2, -1, -1):
        heapify(array, n, i)
    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        heapify(array, i, 0)
    return array


def read_file(filename):
    with open(filename, 'r') as file:
        arr = [int(line.strip()) for line in file]
    return arr


def write_results(filename, results):
    with open(filename, 'w') as file:
        for key, value in results.items():
            file.write(f"{key} {', '.join(str(x) for x in value)}\n")


results = {'Bubble_Sort': [], 'Selection_Sort': [], 'Insertion_Sort': [], 'Shell_Sort': [], 'Merge_Sort': [],
           'Quick_Sort': [], 'Heap_Sort': []}

file_sizes = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
results = {'Bubble_Sort': [], 'Selection_Sort': [], 'Insertion_Sort': [], 'Shell_Sort': [], 'Merge_Sort': [], 'Quick_Sort': [], 'Heap_Sort': []}

for i in file_sizes:
    filename = f"C:\\Users\\mfb36\\Desktop\\A\\file{i}.txt"
    array = read_file(filename)

    for sort_name, sort_func in [('Bubble_Sort', bubble_sort), ('Selection_Sort', selection_sort), ('Insertion_Sort', insertion_sort), ('Shell_Sort', shell_sort), ('Merge_Sort', mergeSort), ('Quick_Sort', quickSort), ('Heap_Sort', heap_sort)]:
        start_time = time.time()
        sort_func(array.copy())
        end_time = time.time()
        results[sort_name].append(round(end_time - start_time, 6))

write_results('sortStats.txt', results)

########################################################################################
#QUESTION III
########################################################################################
print("\n")
print("SOLUTION OF QUESTION III:")
print("****************************")


import random
import time

def sequential_search(lst, key):
    if key in lst:
        return lst.index(key)
    else:
        return -1


def binary_search(lst, key):
    def search(low, high):
        if low > high:
            return -1
        mid = (low + high) // 2
        if lst[mid] == key:
            return mid
        elif lst[mid] < key:
            return search(mid + 1, high)
        else:
            return search(low, mid - 1)
    return search(0, len(lst) - 1)


class HashTable:
    def __init__(self, size):
        self.size = size
        self.slots = [None] * size
        self.data = [None] * size

    def put(self, key, data):
        hash_value = self.hash_function(key)
        if self.slots[hash_value] is None:
            self.slots[hash_value] = key
            self.data[hash_value] = data
        elif self.slots[hash_value] == key:
            self.data[hash_value] = data  # replace
        else:
            next_slot = self.rehash(hash_value)
            while self.slots[next_slot] is not None and self.slots[next_slot] != key:
                next_slot = self.rehash(next_slot)
            if self.slots[next_slot] is None:
                self.slots[next_slot] = key
                self.data[next_slot] = data
            else:
                self.data[next_slot] = data  # replace

    def get(self, key):
        start_slot = self.hash_function(key)
        position = start_slot
        while self.slots[position] is not None:
            if self.slots[position] == key:
                return self.data[position]
            position = self.rehash(position)
            if position == start_slot:
                break
        return None

    def hash_function(self, key):
        return key % self.size

    def rehash(self, old_hash):
        return (old_hash + 1) % self.size



filename = "C:\\Users\\mfb36\\Desktop\\A\\file50000.txt"

with open(filename) as f:
    lst = list(map(int, f.readlines()))

start = time.time()
lst.sort()
end = time.time()
print("Sorting time:", end - start)

ht = {key: "Data" + str(key) for key in lst}

random_numbers = [random.randint(0, 50000) for i in range(1000)]


sequential_search_time = 0
binary_search_time = 0
hashing_time = 0

def measure_time(function, *args):
    start = time.time()
    function(*args)
    end = time.time()
    return end - start

sequential_search_time = sum(measure_time(sequential_search, lst, num) for num in random_numbers)
binary_search_time = sum(measure_time(binary_search, lst, num) for num in random_numbers)
hashing_time = sum(measure_time(ht.get, num) for num in random_numbers)

with open("searchStats.txt", "w") as f:
    f.write("Sequential_Search " + str(sequential_search_time) + "\n")
    f.write("Binary_Search " + str(binary_search_time) + "\n")
    f.write("Hashing " + str(hashing_time) + "\n")

print("Sequential Search Time:", sequential_search_time)
print("Binary Search Time:", binary_search_time)
print("Hashing Time:", hashing_time)



########################################################################################
#QUESTION IV
########################################################################################
print("\n")
print("SOLUTION OF QUESTION IV:")
print("****************************")


import csv
import matplotlib.pyplot as plt

file_stats = {}

with open("C:\\Users\\mfb36\\Desktop\\fileStats.txt") as f:
    for row in csv.reader(f, delimiter=' '):
        file_stats[row[0]] = float(row[1])

with open("C:\\Users\\mfb36\\Desktop\\sortStats.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    sort_stats = {}
    for row in reader:
        sort_stats[row[0]] = [(x) for x in row[1:]]

with open("C:\\Users\\mfb36\\Desktop\\searchStats.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    search_stats = {}
    for row in reader:
        search_stats[row[0]] = float(row[1])


x = range(1, 8)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
fig.tight_layout(pad=1.0)


axs[0].set_title("Performance of Algorithm ")
axs[0].set_xlabel("Input Size MB")
axs[0].set_ylabel("Seconds Passed")
for sort_name, sort_values in sort_stats.items():
    axs[0].plot(x, sort_values, label=sort_name)
for file_name, file_value in file_stats.items():
    axs[0].plot([x[0]], [file_value], '-', label=file_name)
axs[0].legend()

axs[1].set_title("Analysis Performance of Algorithm")
axs[1].set_xlabel("Analysis Algorithm")
axs[1].set_ylabel("Seconds Passed")
axs[1].bar(search_stats.keys(), search_stats.values())

fig.savefig("algorithm_performance_chart.png")
fig.savefig("search_algorithm_performance_chart.png")

plt.show()

########################################################################################
#QUESTION V
########################################################################################
print("\n")
print("SOLUTION OF QUESTION V:")
print("****************************")


class HashTable:
    LOAD_FACTOR_THRESHOLD = 0.7
    INITIAL_SIZE = 11
    RESIZE_FACTOR = 2

    def __init__(self, size=INITIAL_SIZE):
        self.size = size
        self.slots = [None] * self.size
        self.data = [None] * self.size
        self.count = 0

    def put(self, key, data):
        hashvalue = self._hash_function(key)

        if self.slots[hashvalue] is None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data
            self.count += 1
        elif self.slots[hashvalue] == key:
            self.data[hashvalue] = data  # replace
        else:
            nextslot = self._rehash(hashvalue)
            while self.slots[nextslot] is not None and self.slots[nextslot] != key:
                nextslot = self._rehash(nextslot)

            if self.slots[nextslot] is None:
                self.slots[nextslot] = key
                self.data[nextslot] = data
                self.count += 1
            else:
                self.data[nextslot] = data  # replace

        load_factor = self.count / self.size
        if load_factor > self.LOAD_FACTOR_THRESHOLD:
            self._resize()

    def _hash_function(self, key):
        sum = 0
        for pos in range(len(key)):
            sum += (pos + 1) * ord(key[pos])
        return sum % self.size

    def _rehash(self, oldhash):
        # quadratic probing
        return (oldhash + 1 ** 2) % self.size

    def _resize(self):
        oldslots = self.slots
        olddata = self.data

        self.size *= self.RESIZE_FACTOR
        self.slots = [None] * self.size
        self.data = [None] * self.size
        self.count = 0

        for i in range(len(oldslots)):
            if oldslots[i] is not None:
                self.put(oldslots[i], olddata[i])

    def get(self, key):
        startslot = self._hash_function(key)

        data = None
        stop = False
        found = False
        position = startslot
        while self.slots[position] is not None and not found and not stop:
            if self.slots[position] == key:
                found = True
                data = self.data[position]
            else:
                position = self._rehash(position)
                if position == startslot:
                    stop = True
        return data

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, data):
        self.put(key, data)

HT = HashTable()
HT['Abandon'] = "Terk Etmek"
HT['Ability'] = "Yetenek"
HT['Able'] = "MUKTEDİR"
HT['Aboard'] = "(bir taşıtın)İÇİNDE OLMAK"
HT['About'] = "1.HAKKINDA 2.YAKLAŞIK OLARAK"
HT['Above'] = "YUKARIDAKİ"
HT['Abroad'] = "YURT DIŞI"
HT['Absence'] = "YOKLUK"
HT['Absent'] = "1.YOK 2.EKSİK"
HT['Absolute'] = "MUTLAK, KESİN"
HT['Absurd'] = "SAÇMA"
HT['Accept'] = "KABUL ETMEK"
HT['Accident'] = "KAZA"
HT['Accommodate'] = "YERLEŞTİRMEK"
HT['Accommodation'] = " KONAKLAMA YER"
HT['Accompany'] = "EŞLİK ETMEK"
HT['According To'] = "GÖRE"
HT['Account'] = "HESAP"
HT['Accurate'] = "DOĞRU, HATASIZ"
HT['Accuse'] = "SUÇLAMAK"
HT['Ache'] = "AĞRI"
HT['Acquaint'] = "TANIMAK,BİLMEK"
HT['Across'] = "1.BİR UÇTAN DİĞERİNE 2.DİĞER TARAFTA"
HT['Act'] = "1.DAVRANIŞ 2.DAVRANMAK"
HT['Active'] = "ETKİN, FAAL"
HT['Actor'] = "ERKEK OYUNCU"
HT['Actress'] = "KADIN OYUNCU"
HT['Actual'] = "GERÇEK"
HT['Add'] = "TOPLAMAK,EKLEMEK"
HT['Address'] = "ADRES"
HT['Administration'] = "İDARE"
HT['Admire'] = "BEĞENMEK,HAYRAN OLMAK"
HT['Admit'] = "1.KABUL ETMEK 2.İZİN VERMEK"
HT['Adult'] = "YETİŞKİN"
HT['Advance'] = "1.İLERİ 2.AVANS"
HT['Advanced'] = "GELİŞMİŞ"
HT['Advantage'] = "AVANTAJ"
HT['Adventure'] = "MACERA"
HT['Advertise'] = "REKLAM YAPMAK, İLAN VERMEK"
HT['Advice'] = "TAVSİYE"
HT['Advise'] = "TAVSİYE ETMEK"
HT['Aerial'] = "ANTEN"
HT['Aeroplane'] = "UÇAK"
HT['Affair'] = "1.OLAY 2.İŞ 3.İLİŞKİ"
HT['Affect'] = "ETKİLEMEK"
HT['Afford'] = "SATIN ALMA GÜCÜ OLMAK"
HT['Afraid'] = "KORKMAK"
HT['After'] = "SONRA"
