# Parsing readme

## API

The parser API consist of four different reading functions.

```python

a = read_task1_csv(<TextFileDescriptor>, grades=<bool>)
b = read_task2_csv(<TextFileDescriptor>, grades=<bool>)

c = read_task1_pb(<BinaryFileDescriptor>)
d = read_task2_pb(<BinaryFileDescriptor>)

```

These four functions allow the user to read csv and protobuf files into the internal format.

It is also possible to write to a protobuf file by using the `Write_PB(<BinaryFileDescriptor>)` function from the `HeadlineCollection` and `CandidateCollection` classes.

The collection classes behave like regular python collections. They can be iterated over and appended to like any other collection.

Here is an example of use:

```python

import parsing

headlinecollection = None

with open("pbfile.bin", "rb") as fd:
    headlinecollection = parsing.read_task1_pb(fd)

for headline in headlinecollection:
    headline.sentence = [word.replace("/", "\\") for word in headline.sentence]

test = parsing.Headline(
    42,
    ["This", "is", "a", "sentence"],
    3,
    "joke",
    [2, 1, 0, 0, 0],
    0.6
)

headlinecollection.append(test)

with open("pbfile_new.bin", "wb") as fd:
    headlinecollection.Write_PB(fd)


```
