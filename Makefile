#!/bin/bash
#
#
mle:
	python3 srkMLE.py

mc:
	python3 srkICaLMarkovChain.py
clp:
	python3 srkpClamp2ascii.py

# this will convert an abf file to plain text. there are 3 columns: time, voltage, current.
abf:
	python3 srkPyABF.py

clean:
	rm -rf sample.txt solution.dat myEstimates.txt
