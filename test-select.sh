#!/bin/bash

IN_CSV="../arxiv-metadata-cs.LG.json"
OUT_CSV="selection.csv"
KEYWORDS='data centric,dcai'


if [ -f "$OUT_CSV" ]; then
    alreview select -i $IN_CSV -o $OUT_CSV -s $OUT_CSV -kw "$KEYWORDS" -v
else 
    alreview select -i $IN_CSV -o $OUT_CSV -kw "$KEYWORDS" -v
fi