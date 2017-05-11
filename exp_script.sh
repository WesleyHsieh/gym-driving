#!/bin/bash
for number in 1 2 3 4 5 6 7 8 9 10 11 12
do
	python test_ray_exp.py --num_workers $number
done
exit 0