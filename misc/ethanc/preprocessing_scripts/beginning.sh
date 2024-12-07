#!/bin/bash
for num in `seq 1 24`
do
	dcm2niix -f '%f_%p' -o /home/ethanchurch/TestData/dcm_converted/ /home/ethanchurch/TestData/19795-20220304-SF_01030/${num}-*
done
