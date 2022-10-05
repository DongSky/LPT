import os

label = []
index_to_label = {}

f = open("resisc45_labels.txt")
label = [c.strip() for c in f.readlines()]
f.close()
for i in range(len(label)):
    index_to_label[i] = label[i]
f = open("resisc45_train.txt")
filenames = [c.strip() for c in f.readlines()]
f.close()
output_filenames = []
for filename in filenames:
    dirs, name = filename.split("/")
    dirs = index_to_label[int(dirs)]
    output_filenames.append(f"{dirs}/{name}")
f = open("resisc45_train.txt", "w")
for i in output_filenames:
    f.write(i+"\n")
f.close()