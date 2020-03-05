import csv
import json
from collections import Iterator
from typing import List, Dict, TextIO, BinaryIO
from enum import Enum

from bidict import bidict
import numpy as np
from tqdm import tqdm

import lib.parsing.Headline_pb2 as Headline_pb2
import lib.parsing.Candidates_pb2 as Candidates_pb2

__all__ = [
    "Label",
    "Headline",
    "HeadlineCollection",
    "Candidates",
    "CandidateCollection",
    "read_task1_csv",
    "read_task1_pb",
    "read_task2_csv",
    "read_task2_pb"
]

class Label(Enum):
    NA = -1
    EQUAL = 0
    H1 = 1
    H2 = 2

LabelTranslate = bidict({
    Label.NA: Candidates_pb2.CandidateCollection.Candidates.Label.NA,
    Label.EQUAL: Candidates_pb2.CandidateCollection.Candidates.Label.EQUAL,
    Label.H1: Candidates_pb2.CandidateCollection.Candidates.Label.H1,
    Label.H2: Candidates_pb2.CandidateCollection.Candidates.Label.H2
})

def LabelToPB(l : Label) -> Candidates_pb2.CandidateCollection.Candidates.Label:
    return LabelTranslate[l]

def PBToLabel(l : Candidates_pb2.CandidateCollection.Candidates.Label) -> Label:
    return LabelTranslate.inverse[l]

class Headline:
    """An object representing a Headline with edit.
    Can optionally hold grading data

    Raises:
        ValueError: Raised on invalid BERT config
    """
    BERT_VOCAB = None
    BERT_VECTOR_LENGTH = 27

    UNKNOWN = 100
    CLS = 101
    SEP = 102

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
            res.extend(feature.compute_feature(self))
        return res

    def GetTokenized(self) -> List[str]:
        return self.sentence

    def GetTokenizedWEdit(self) -> List[str]:
        l = self.sentence
        l[self.word_index] = self.edit
        return l

    @classmethod
    def SetBERTVocab(cls, vocab_fd : TextIO) -> None:
        """Sets the BERT vocab statically.
        This needs to be called before calling GetBERT is called!

        Args:
            vocab_fd (TextIO): A file descriptor pointing to a BERT vocab.
        """
        cls.BERT_VOCAB = {}
        for i, t in enumerate(vocab_fd):
            cls.BERT_VOCAB[t.strip("\n")] = i

    def GenerateBERT(self) -> None:
        if self.BERT_VOCAB is None:
            raise ValueError("Bert Vocab is None: Set it with Headline.SetBERTVocab()")

        self.bert_vector = [0]*self.BERT_VECTOR_LENGTH

        if len(self.sentence) + 2 > self.BERT_VECTOR_LENGTH:
            raise ValueError(f"Headline.BERT_VECTOR_LENGTH is not high enough. A sentence of length {len(self.sentence)} did not fit.")

        for i, word in enumerate(self.GetTokenizedWEdit()):
            if word in self.BERT_VOCAB:
                self.bert_vector[i+1] = self.BERT_VOCAB[word]
            else:
                self.bert_vector[i+1] = self.UNKNOWN

        self.bert_vector[0] = self.CLS
        self.bert_vector[len(self.sentence)+1] = self.SEP

    def GetBERT(self) -> np.ndarray:
        if self.bert_vector is None:
            self.GenerateBERT()
        return np.asarray(self.bert_vector)

    def GetEdited(self) -> str:
        sent = self.sentence
        sent[self.word_index] = self.edit
        return " ".join(sent)

    def ToPB(self, HL : Headline_pb2.HeadlineCollection.Headline) -> None:
        HL.id = self.id
        HL.sentence.extend(self.sentence)
        HL.word_index = self.word_index
        HL.edit = self.edit
        HL.grades.extend(self.grades)
        HL.avg_grade = -1 if self.avg_grade is None else self.avg_grade

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
    """A Collection of Headline objects.
    """
    def __init__(
        self,
        iterable : List[Headline] = None
    ):
        self.collection = iterable if iterable else []

    def append(self, H : Headline) -> None:
        self.collection.append(H)

    def extend(self, HC) -> None:
        self.collection.extend(HC.collection)

    def AddFeature(self, feature) -> None:
        for e in self.collection:
            e.AddFeature(feature)

    def AddFeatures(self, features : List) -> None:
        for e in self.collection:
            e.AddFeatures(features)

    def GetFeatureVectors(self) -> np.ndarray:
        features = []
        for e in tqdm(self.collection, unit='headline(s)', desc="Computing features"):
            features.append(e.GetFeatureVector())
        
        return np.asarray(features, dtype='float32')

    def GetIDs(self) -> np.ndarray:
        return np.array(
            [h.id for h in self.collection]
        )

    def GetBERT(self) -> np.ndarray:
        return np.asarray(
            [h.GetBERT() for h in self.collection]
        )

    def GetGrades(self) -> np.ndarray:
        return np.array(
            [h.avg_grade for h in self.collection]
        )

    def GetEditSentences(self) -> np.ndarray:
        return np.array(
            [h.GetEdited() for h in self.collection]
        )

    def GetSentences(self) -> np.ndarray:
        return np.array(
            [" ".join(h.sentence) for h in self.collection]
        )

    def GetTokenizedWEdit(self) -> np.ndarray:
        return np.array(
            [h.GetTokenizedWEdit() for h in self.collection]
        )

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
    """An object representing a pair of Headline with an optional Label
    """
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

    def GetFeatureVectors(self) -> np.ndarray:
        return np.array([
            self.HL1.GetFeatureVector(),
            self.HL2.GetFeatureVector()
        ])

    def GetTokenizedWEdit(self) -> np.ndarray:
        arr = [
            self.HL1.GetTokenizedWEdit(),
            self.HL2.GetTokenizedWEdit()
        ]
        return arr

    def GetEditedSentences(self) -> np.ndarray:
        return np.array([
            self.HL1.GetEdited(),
            self.HL2.GetEdited()
        ])

    def GetIDS(self) -> np.ndarray:
        return f'{self.HL1.id}-{self.HL2.id}'

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
    """A collection of Candidate objects.
    """
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
            e.AddFeatures(features)

    def GetFeatureVectors(self) -> np.ndarray:
        return np.array(
            [c.GetFeatureVectors() for c in self.collection]
        )

    def GetTokenizedWEdit(self) -> List:
        return [c.GetTokenizedWEdit() for c in self.collection]

    def GetEditedSentences(self) -> np.ndarray:
        return np.array(
            [c.GetEditedSentences() for c in self.collection]
        )

    def GetIDS(self) -> np.ndarray:
        return np.array(
            [cc.GetIDS() for cc in self.collection]
        )

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

    s = st.strip().split(" ")
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

def read_task1_pb(fd : TextIO) -> HeadlineCollection:
    res = HeadlineCollection()
    hlc_pb = Headline_pb2.HeadlineCollection()
    hlc_pb.ParseFromString(fd.read())
    res.FromPB(hlc_pb)
    del hlc_pb
    return res

def read_task2_pb(fd : TextIO) -> CandidateCollection:
    res = CandidateCollection()
    clc_pb = Candidates_pb2.CandidateCollection()
    clc_pb.ParseFromString(fd.read())
    res.FromPB(clc_pb)
    del clc_pb
    return res
