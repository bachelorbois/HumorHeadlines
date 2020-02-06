import csv
import json
from collections import Iterator
from typing import List, Dict, TextIO, BinaryIO
from enum import Enum

import numpy as np

import lib.parsing.Headline_pb2 as Headline_pb2
import lib.parsing.Candidates_pb2 as Candidates_pb2


class Label(Enum):
    NA = -1
    EQUAL = 0
    H1 = 1
    H2 = 2

def LabelToPB(l : Label) -> Candidates_pb2.CandidateCollection.Candidates.Label:
    if l == Label.NA: return Candidates_pb2.CandidateCollection.Candidates.Label.NA
    if l == Label.EQUAL: return Candidates_pb2.CandidateCollection.Candidates.Label.EQUAL
    if l == Label.H1: return Candidates_pb2.CandidateCollection.Candidates.Label.H1
    if l == Label.H2: return Candidates_pb2.CandidateCollection.Candidates.Label.H2
    raise ValueError()

def PBToLabel(l : Candidates_pb2.CandidateCollection.Candidates.Label) -> Label:
    if l == Candidates_pb2.CandidateCollection.Candidates.Label.NA: return Label.NA
    if l == Candidates_pb2.CandidateCollection.Candidates.Label.EQUAL: return Label.EQUAL
    if l == Candidates_pb2.CandidateCollection.Candidates.Label.H1: return Label.H1
    if l == Candidates_pb2.CandidateCollection.Candidates.Label.H2: return Label.H2
    raise ValueError()

class Headline:
    BERT_VOCAB = None
    BERT_VECTOR_LENGTH = 20

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
        self.features = []
        self.bert_vector = None

    def AddFeature(self, feature) -> None:
        self.features.append(feature)

    def AddFeatures(self, features : List) -> None:
        self.features.extend(features)

    def GetFeatureVector(self) -> np.ndarray:
        res = []
        for feature in self.features:
            res.extend(
                feature.compute_feature(self)
            )

        return np.array(res)

    def GetTokenized(self) -> List[str]:
        return self.sentence

    def GetTokenizedWEdit(self) -> List[str]:
        l = self.sentence
        l[self.word_index] = self.edit
        return l

    @classmethod
    def SetBERTVocab(cls, vocab_fd : TextIO) -> None:
        cls.BERT_VOCAB = {}
        for i, t in enumerate(vocab_fd):
            cls.BERT_VOCAB[t] = i

    def GenerateBERT(self) -> None:
        if self.BERT_VOCAB is None:
            raise ValueError("Bert Vocab is None: Set it with Headline.SetBERTVocab()")

        self.bert_vector = [0]*self.BERT_VECTOR_LENGTH

        for i, word in enumerate(self.sentence):
            if word in self.BERT_VOCAB:
                self.bert_vector[i] = self.BERT_VOCAB[word]
            else:
                self.bert_vector[i] = 100

    def GetBERT(self) -> np.ndarray:
        if self.bert_vector is None:
            self.GenerateBERT()
        return np.array(self.bert_vector)

    def ToPB(self, HL : Headline_pb2.HeadlineCollection.Headline) -> None:
        HL.id = self.id
        HL.sentence.extend(self.sentence)
        HL.word_index = self.word_index
        HL.edit = self.edit
        HL.grades.extend(self.grades)
        HL.avg_grade = self.avg_grade if self.avg_grade else -1

    def ToDict(self) -> Dict:
        return {
            "id": self.id,
            "sentence": " ".join(self.sentence),
            "word_index": self.word_index,
            "edit": self.edit,
            "grades": self.grades,
            "avg_grade": self.avg_grade
        }

    def __len__(self) -> int:
        return len(self.sentence)

    def __str__(self) -> str:
        return json.dumps(self.ToDict(), indent=4)

class HeadlineCollection:
    def __init__(
        self,
        iterable : List[Headline] = None
    ):
        self.collection = iterable if iterable else []

    def append(self, H : Headline) -> None:
        self.collection.append(H)

    def AddFeature(self, feature) -> None:
        for e in self.collection:
            e.AddFeature(feature)

    def AddFeatures(self, features : List) -> None:
        for e in self.collection:
            e.AddFeatures(features)

    def ToPB(self) -> Headline_pb2.HeadlineCollection:
        col_pb = Headline_pb2.HeadlineCollection()
        for HL in self.collection:
            HL_pb = col_pb.HL.add()
            HL.ToPB(HL_pb)
        return col_pb

    def FromPB(self, pb : Headline_pb2.HeadlineCollection) -> None:
        self.collection = []
        for HL in pb.HL:
            self.collection.append(
                build_headline_pb(HL)
            )

    def Write_PB(self, fd : BinaryIO) -> None:
        fd.write(self.ToPB().SerializeToString())

    def __iter__(self) -> None:
        for e in self.collection:
            yield e

    def __getitem__(self, index : int):
        return self.collection[index]

    def __len__(self):
        return len(self.collection)

    def __str__(self) -> str:
        return json.dumps(
            [e.ToDict() for e in self.collection],
            indent=4
        )


class Candidates:
    def __init__(
        self,
        headline1 : Headline,
        headline2 : Headline,
        label : Label
    ):
        self.HL1 = headline1
        self.HL2 = headline2
        self.label = label

    def AddFeature(self, feature) -> None:
        self.HL1.AddFeature(feature)
        self.HL2.AddFeature(feature)

    def AddFeatures(self, features : List) -> None:
        self.HL1.AddFeatures(features)
        self.HL2.AddFeatures(features)

    def ToPB(self, C : Candidates_pb2.CandidateCollection.Candidates) -> None:
        self.HL1.ToPB(C.HL1)
        self.HL2.ToPB(C.HL2)
        C.label = LabelToPB(self.label)

    def ToDict(self) -> Dict:
        return {
            "HL1": self.HL1.ToDict(),
            "HL2": self.HL2.ToDict(),
            "label": self.label.name,
        }

    def __str__(self) -> str:
        return json.dumps(self.ToDict(), indent=4)

class CandidateCollection:
    def __init__(
        self,
        iterable : List[Candidates] = None
    ):
        self.collection = iterable if iterable else []

    def append(self, H : Candidates) -> None:
        self.collection.append(H)

    def AddFeature(self, feature) -> None:
        for e in self.collection:
            e.AddFeature(feature)

    def AddFeatures(self, features : List) -> None:
        for e in self.collection:
            e.AddFeatures(feature)

    def ToPB(self) -> Candidates_pb2.CandidateCollection:
        col_pb = Candidates_pb2.CandidateCollection()
        for cand in self.collection:
            cand_pb = col_pb.candidates.add()
            cand.ToPB(cand_pb)
        return col_pb

    def FromPB(self, pb : Candidates_pb2.CandidateCollection) -> None:
        self.collection = []
        for cand in pb.candidates:
            self.collection.append(
                build_candidates_pb(cand)
            )

    def Write_PB(self, fd : BinaryIO) -> None:
        fd.write(self.ToPB().SerializeToString())

    def __iter__(self) -> None:
        for e in self.collection:
            yield e

    def __getitem__(self, index : int):
        return self.collection[index]

    def __len__(self):
        return len(self.collection)

    def __str__(self) -> str:
        return json.dumps(
            [e.ToDict() for e in self.collection],
            indent=4
        )

def build_headline(l : List, grades = True) -> Headline:
    st = l[1]
    MASKING_PHRASE = "¤^~"
    try:
        a = st.index("<")
        b = st.index(">")

        st = st[:a] + st[a:b+1].replace(" ", MASKING_PHRASE) + st[b+1:]

    except ValueError:
        raise ValueError("Sentence did not contain a tagged word")

    s = st.split(" ")
    ind = -1

    for i, e in enumerate(s):
        if "/>" in e:
            s[i] = s[i].replace(MASKING_PHRASE, " ")
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

def build_headline_pb(pb : Headline_pb2.HeadlineCollection.Headline) -> Headline:
    return Headline(
        pb.id,
        [w for w in pb.sentence],
        pb.word_index,
        pb.edit,
        [g for g in pb.grades],
        pb.avg_grade if pb.avg_grade!=-1 else None
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

def build_candidates_pb(pb : Candidates_pb2.CandidateCollection.Candidates) -> Candidates:
    return Candidates(
        build_headline_pb(pb.HL1),
        build_headline_pb(pb.HL2),
        PBToLabel(pb.label)
    )

def read_task1_csv(fd : TextIO, grades = True) -> HeadlineCollection:
    res = HeadlineCollection()
    csv_reader = csv.reader(fd, delimiter=',', quotechar='"', )
    next(csv_reader)
    for row in csv_reader:
        res.append(build_headline(row, grades))

    return res

def read_task2_csv(fd : TextIO, grades = True) -> CandidateCollection:
    res = CandidateCollection()
    csv_reader = csv.reader(fd, delimiter=',', quotechar='"', )
    next(csv_reader)
    for row in csv_reader:
        res.append(build_candidates(row, grades))

    return res

def read_task1_pb(fd : TextIO, grades = True) -> HeadlineCollection:
    res = HeadlineCollection()
    hlc_pb = Headline_pb2.HeadlineCollection()
    hlc_pb.ParseFromString(fd.read())
    res.FromPB(hlc_pb)
    del hlc_pb
    return res

def read_task2_pb(fd : TextIO, grades = True) -> CandidateCollection:
    res = CandidateCollection()
    clc_pb = Candidates_pb2.CandidateCollection()
    clc_pb.ParseFromString(fd.read())
    res.FromPB(clc_pb)
    del clc_pb
    return res
