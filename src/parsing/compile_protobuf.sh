#!/bin/bash

protoc --python_out=./ ./Headline.proto
protoc --python_out=./ ./Candidates.proto
