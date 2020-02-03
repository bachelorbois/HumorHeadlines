import sys
sys.path.append("..")
import lib.parsing

ins = [
    "../../data/task-1/preproc/1_original_train.bin",
    "../../data/task-1/preproc/1_original_dev.bin",
    "../../data/task-2/preproc/1_original_train.bin",
    "../../data/task-2/preproc/1_original_dev.bin"
]

read_funcs = [
    parsing.read_task1_pb,
    parsing.read_task1_pb,
    parsing.read_task2_pb,
    parsing.read_task2_pb
]

outs = [
    "../../data/task-1/preproc/2_concat_train.bin",
    "../../data/task-1/preproc/2_concat_dev.bin",
    "../../data/task-2/preproc/2_concat_train.bin",
    "../../data/task-2/preproc/2_concat_dev.bin"
]

def proc(i, o, f):
    with open(i, "rb") as fd:
        a = f(fd)


        def prochl(hl):
            dels = []
            for i, w in enumerate(hl.sentence):
                if len(w) == 0:
                    continue
                if w[0] == "'":
                    if i != 0:
                        if len(w) != 1 or hl.sentence[i-1][-1] == "s":
                            hl.sentence[i-1] += w
                            dels.append(i)

            for i in sorted(dels, reverse=True):
                del hl.sentence[i]
            before = len(list(filter(
                lambda x: x<hl.word_index,
                dels
            )))

            hl.word_index -= before

        if isinstance(a, parsing.CandidateCollection):
            for c in a:
                prochl(c.HL1)
                prochl(c.HL2)
        else:
            for hl in a:
                prochl(hl)


        with open(o, "wb") as fd:
            a.Write_PB(fd)
    del a

for i in range(len(ins)):
    proc(ins[i], outs[i], read_funcs[i])
