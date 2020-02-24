import sys
sys.path.append("..")
import lib

ins = [
    "../../data/task-1/dev.csv",
    "../../data/task-1/test.csv",
    "../../data/task-2/dev.csv",
    "../../data/task-2/test.csv"
]

grades = [
    True,
    False,
    True,
    False
]

read_funcs = [
    lib.read_task1_csv,
    lib.read_task1_csv,
    lib.read_task2_csv,
    lib.read_task2_csv
]

outs = [
    "../../data/task-1/preproc/1_original_dev.bin",
    "../../data/task-1/preproc/1_original_test.bin",
    "../../data/task-2/preproc/1_original_dev.bin",
    "../../data/task-2/preproc/1_original_test.bin"
]

def proc(i, o, g, f):
    with open(i, "r") as fd:
        a = f(fd, g)
        with open(o, "wb") as fd:
            a.Write_PB(fd)
    del a

for i in range(len(ins)):
    proc(ins[i], outs[i], grades[i], read_funcs[i])

# FunLines
proc(
    "../../data/FunLines/task-1/train_funlines.csv",
    "../../data/FunLines/task-1/preproc/1_original_train.bin",
    True,
    lib.read_task1_csv
)

proc(
    "../../data/FunLines/task-2/train_funlines.csv",
    "../../data/FunLines/task-2/preproc/1_origial_train.bin",
    True,
    lib.read_task2_csv
)
