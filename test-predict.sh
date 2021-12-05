#!/bin/bash

INPUT_FILE="../arxiv-metadata-cs.LG.json"
ORACLE_FILE="selection.csv"
OUTPUT_FILE="prediction.csv"


alreview predict -i $INPUT_FILE -O $ORACLE_FILE -o $OUTPUT_FILE -t 0.2