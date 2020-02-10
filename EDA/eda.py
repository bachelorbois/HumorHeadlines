import numpy as np
import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re



def task_1():

    header_types = {'id': str, 'original': str, 'edit': str, 'grades': str, 'meanGrade':np.float64}

    data = pd.read_csv("data/task-1/train.csv", delimiter=",", dtype=header_types).values

    grades = data[:,-1]
    print("Mean grade task 1: ", round(np.mean(grades),2))
    grades = Counter(grades)

    plt.figure(0)
    plt.bar(grades.keys(), grades.values(), width=0.1)
    plt.title("Task 1 - train - Grades disribution")
    plt.savefig("plots/task-1-train - grades distrib.png")

    edits = data[:,2]
    edit_freq = Counter(edits).most_common(20)
    keys, values = [], []
    for word in edit_freq:
        keys.append(word[0])
        values.append(word[1])

    plt.figure(1)
    plt.title("Task 1 - train - Most common edits words")
    plt.bar(keys, values, width=0.6)
    plt.xticks(rotation=90)
    plt.savefig("plots/task-1-train - common edit words.png")

    entity = []
    tags = nltk.pos_tag(edits)
    for tag in tags:
        entity.append(tag[1])

    entity = Counter(entity, sorted=True).most_common(20)
    keys, values = [], []
    for pos_tag in entity:
        keys.append(pos_tag[0])
        values.append(pos_tag[1])
    plt.figure(5)
    plt.title("Task 1 - train - Most common edits POS")
    plt.bar(keys, values, width=0.4)
    plt.xticks(rotation=90)
    plt.savefig("plots/task-1-train - common replacements.png")

    

def task_2():

    header_types = {'id': str, 'original1': str, 'edit1': str, 'grades1': str, 'meanGrade1':np.float64, 
                    'original2': str, 'edit2': str, 'grades2': str, 'meanGrade2':np.float64, 'label':np.int32}

    data = pd.read_csv("data/task-2/train.csv", delimiter=",", dtype=header_types).values

    grades = data[:,4]
    print("Mean grade task 2 - edit 1: ", round(np.mean(grades),2))
    grades = Counter(grades)
    plt.figure(3)
    plt.title("Task 2 - train - Grades disribution Edit 1")
    plt.bar(grades.keys(), grades.values(), width=0.1)
    plt.savefig("plots/task-2-train - grades distrib edit 1.png")

    grades = data[:,8]
    print("Mean grade task 2 - edit 2: ", round(np.mean(grades),2))
    grades = Counter(grades)
    plt.figure(4)
    plt.title("Task 2 - train - Grades disribution Edit 2")
    plt.bar(grades.keys(), grades.values(), width=0.1)
    plt.savefig("plots/task-2-train - grades distrib edit 2.png")

def check_multi_token_entity_task_1(file):
    header_types = {'id': str, 'original': str, 'edit': str, 'grades': str, 'meanGrade':np.float64}
    data = pd.read_csv(file, delimiter=",", dtype=header_types).values

    edits = data[:,2]
    for row, edit in enumerate(edits):
        words = edit.split(" ")
        if len(words) != 1:
            pass
            #print("ID: "+str(i)+" WORDS:", words)

    original = data[:,1]
    
    to_replace = []
    for sent in original:
        entity = re.finditer(r'\<.*?\>', sent)
        for item in entity:
            to_replace.append(item.group(0).replace("<", "").replace("/>", ""))

    count = 0
    for i, repl in enumerate(to_replace):
        words = repl.split(" ")
        if len(words) > 2:
            print("ID: "+str(i)+" WORDS:", words)
            count = count+1
    print("Number of samples with multi word entity in file "+file+": ", count)


def check_multi_token_entity_task_2(file):
    header_types = {'id': str, 'original1': str, 'edit1': str, 'grades1': str, 'meanGrade1':np.float64, 
                    'original2': str, 'edit2': str, 'grades2': str, 'meanGrade2':np.float64, 'label':np.int32}
    data = pd.read_csv(file, delimiter=",", dtype=header_types).values

    original = data[:,1]
    
    to_replace = []
    for sent in original:
        entity = re.finditer(r'\<.*?\>', sent)
        for item in entity:
            to_replace.append(item.group(0).replace("<", "").replace("/>", ""))

    count = 0
    for i, repl in enumerate(to_replace):
        words = repl.split(" ")
        if len(words) > 2:
            #print("ID: "+str(i)+" WORDS:", words)
            count = count+1
    print("Number of samples with multi word entity in file "+file+": ", count)

    

if __name__ == "__main__":

    task_1()
    task_2()
    check_multi_token_entity_task_1("data/task-1/train.csv")
    check_multi_token_entity_task_1("data/task-1/dev.csv")
    check_multi_token_entity_task_2("data/task-2/train.csv")
    check_multi_token_entity_task_2("data/task-2/dev.csv")
    

