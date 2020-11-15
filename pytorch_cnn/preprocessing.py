#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:18:50 2020

@author: briag
"""

import os
import csv
from random import choice, sample, randint
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def filenames(folder):
    """makes a list of file names from a folder
    
    Inputs:
    ----------
    folder: path of folde
    
    Returns
    -------
    filenanmes: list of filenames
    """
    filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return filenames

def display_images(
    folder: str,
    filenames: str, 
    columns=5, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):

    if not filenames:
        print("No images to display.")
        return 

    files_draw = sample(filenames, k=max_images)
    images = [Image.open(folder + "/" + f) for f in files_draw]
    height = max(height, int(len(files_draw)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, file in enumerate(files_draw):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(Image.open(folder + "/" + file))
        plt.title(file, fontsize=label_font_size);
    plt.show()
        
def rand_pic_label(pix_folder, files_labels_dict, labels):
    pic_name = choice(list(files_labels_dict.keys()))
    label = ""
    for i in range(len(labels)):
        if files_labels_dict[pic_name][i] == 1:
            label += " " + labels[i] + ","
    plt.imshow(Image.open(pix_folder + "/" + pic_name))
    plt.title(pic_name +" Categories:" + label)
    plt.show(); 
        
def image_sizes(folder, filenames):
    print(f"There are {len(filenames)} pictures.")
    sizes = []
    for file in filenames:
        img = Image.open(folder + "/" + file)
        sizes.append(img.size)
        uniques = Counter(sizes)
    for c in uniques:
        print(f"Count of size {c}: {uniques[c]}")
        
def resize_pix(folder,filenames,targert_folder,size):
    if not os.path.exists(targert_folder):
        os.makedirs(targert_folder)
    resized = []
    for f in filenames:
        
        try:
            img = Image.open(folder + "/" + f)
            img = img.resize((size,size))
            resized.append(f)
            img.save(targert_folder + "/" + f)
        except:
            print(f"Could not resize: {f}")
            os.remove(targert_folder + "/" + f)
            resized.remove(f)
    return resized

def files_labels_todict(folder, txt_file):
    with open(folder + '/' + txt_file, newline='') as f:
        reader = csv.reader(f, delimiter=' ')
        files_labels = {}
        for row in reader:
            files_labels[row[0]] = list(map(int, row[1:]))
    return files_labels


    
def category_count(files_labels_dict,labels):
    cat_count = {}
    counts = [0 for _ in labels]
    for val in files_labels_dict.values():
        for i in range(len(labels)):
            counts[i] += val[i]
    for i in range(len(labels)):
        cat_count[labels[i]] = counts[i]
    return cat_count

def main():
    train_folder = "Challenge_train/train"
    train_files = filenames(train_folder)
    display_images(train_folder, train_files)
    
    print("Raw Images info:")
    image_sizes(train_folder, train_files)
    
    print("Resizing images:")
    train_resize_folder = "train_resized"
    train_resized = resize_pix(train_folder, train_files, train_resize_folder,480)
    image_sizes(train_resize_folder, train_resized)
    
    display_images(train_resize_folder, train_resized)
    
    files_labels_dict = files_labels_todict("Challenge_train", "train.anno.txt")
    labels = ["indoor", "outdoor", "person", "day", "night", "water", "road", "vegetation", "tree", "mountains", "beach", "buildings", "sky", "sunny", "partly_cloudy", "overcast", "animal"]
    cat_count = category_count(files_labels_dict,labels)
    
    test_files_labels_dict = files_labels_todict("Challenge_test", "test.anno.txt")
    test_cat_count = category_count(test_files_labels_dict,labels)
    
    rand_pic_label(train_resize_folder, files_labels_dict, labels)
    
    train_resize_folder = "train_resized"
    return test_cat_count, cat_count
    
test_cat_count, cat_count = main()