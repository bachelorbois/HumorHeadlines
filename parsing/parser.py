import csv
import json
from typing import List, Dict, TextIO
from enum import Enum

class Label(Enum):
    NA = -1
    EQUAL = 0
    H1 = 1
    H2 = 2

class Headline:
    def __init__(
        self,
        id : int,
        sentence : List[str],
        word_index : int,
        edit : str,
        grades : List[int] = None,
        avg_grade : float = None
    ):
        self.id = id
        self.sentence = sentence
        self.word_index = word_index
        self.edit = edit
        self.grades = grades
        self.avg_grade = avg_grade

    def GetTokenized(self) -> List[str]:
        return self.sentence

    def GetTokenizedWEdit(self) -> List[str]:
        l = self.sentence
        l[self.word_index] = self.edit
        return l

    def ToDict(self) -> Dict:
        return {
            "id": self.id,
            "sentence": " ".join(self.sentence),
            "word_index": self.word_index,
            "edit": self.edit,
            "grades": self.grades,
            "avg_grade": self.avg_grade
        }

    def __str__(self) -> str:
        return json.dumps(self.ToDict(), indent=4)

class Candidates:
    def __init__(
        self,
        headline1 : Headline,
        headline2 : Headline,
        label : Label
    ):
        self.H1 = headline1
        self.H2 = headline2
        self.label = label

    def ToDict(self) -> Dict:
        return {
            "H1": self.H1.ToDict(),
            "H2": self.H2.ToDict(),
            "label": self.label.name,
        }

    def __str__(self) -> str:
        return json.dumps(self.ToDict(), indent=4)

def build_headline(l : List, grades = True) -> Headline:
    s = l[1].split(" ")
    ind = -1
    for i, e in enumerate(s):
        if "/>" in e:
            ind = i
            break

    s[ind] = s[ind].replace("<", "").replace("/>", "")

    return Headline(
        int(l[0]),
        s,
        ind,
        l[2],
        [int(c) for c in l[3]] if grades else None,
        float(l[4]) if grades else None
    )

def build_candidates(l : List, grades = True) -> Candidates:
    ids = [int(e) for e in l[0].split("-")]
    ls = [
        l[1:5] if grades else l[1:3],
        l[5:9] if grades else l[3:6]
    ]

    return Candidates(
        build_headline([ids[0]] + ls[0], grades),
        build_headline([ids[1]] + ls[1], grades),
        Label(int(l[-1])) if grades else Label.NA
    )

def read_task1(fd : TextIO, grades = True) -> List[Headline]:
    res = []
    csv_reader = csv.reader(fd, delimiter=',', quotechar='"', )
    next(csv_reader)
    for row in csv_reader:
        res.append(build_headline(row, grades))

    return res

def read_task2(fd : TextIO, grades = True) -> List[Headline]:
    res = []
    csv_reader = csv.reader(fd, delimiter=',', quotechar='"', )
    next(csv_reader)
    for row in csv_reader:
        res.append(build_candidates(row, grades))

    return res
