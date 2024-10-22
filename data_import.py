import os
import lzma
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures


# Creating a function to import files from folder
def xz_files_in_dir(directory):
    files = []  # Initialising the files list
    for filename in os.listdir(directory):  # Looping over all files in the folder
        if filename.endswith('.xz'):  # Check if the file is an .xz file
            files.append(filename)  # Appending our files list
    return files

    

folder_path = "/Users/scottpitcher/Desktop/python/Github/Data/" # Directory of the folder holding our .xz files
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = 'vocab.txt'

# Loop to obtain desired train amount
while True:
    try:
        split_prcnt = int(input("What percent of the data would you like to use for training? (0-100): "))/100
        if 0 <= split_prcnt <= 100:
            break  # If valid input, exit the loop
        else:
            print("Please enter a value between 0 and 100.")
    except ValueError:  # In case a non-integer value is entered
        print("Please enter a valid integer.")

files = xz_files_in_dir(folder_path)
total_files = len(files)
print(total_files)

split_index = int(total_files * split_prcnt)  # 90% for training

files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()



with open(output_file_train,"w", encoding='utf-8') as outfile: # Opening the output file
    for filename in tqdm(files_train, total = len(files_train)): # Looping over every file until max_count reached
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(output_file_val,"w", encoding='utf-8') as outfile: # Opening the output file
    for filename in tqdm(files_val, total = len(files_val)): # Looping over every file until max_count reached
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Writing all of our vocabulary to our Vocab file
with open(vocab_file, 'w', encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')