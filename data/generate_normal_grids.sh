#! /usr/bin/bash

Rscript vquantization.r $1 $2 $3
python export.py
