import os
import lzma
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures


# Creating a function to import files from folder
def xz_files_in_dir(directory):
    files =[] # initialising the files
    for filename in os.listdir(directory): #Looping over all files in the folder
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.xz') : #Ensuring we are retreiving the correct files
            files.append(filename) # Appending our files list
        return files
    

folder_path = "/Users/scottpitcher/Desktop/python/Github/Data" # Directory of the folder holding our .xz files
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = 'vocab.txt'

files = xz_files_in_dir(folder_path)
total_files = len(files)

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

split_index = int(total_files * split_prcnt)  # 90% for training

files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()



with open(output_file_train.format(i),"w", encoding='utf-8') as outfile: # Opening the output file
    for count, filename in enumerate(tqdm(files[:max_count], total = max_count)): # Looping over every file until max_count reached
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)
    files = files[max_count:]

# Writing all of our vocabulary to our Vocab file
with open(vocab_file, 'w', encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')