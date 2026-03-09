"""
created by jwenner on 5.29.2025
goal is to remove as many aimpoints before and after center as possible to get a reduced number of aimpoints
"""

import math
cycle_count = 0
aim_list = list(range(1,65))
num_sxns = 8
a_cent = math.ceil(len(aim_list) / 2) # the number!
incr=0
flip=1
# find the index of the center:
print(aim_list)
while len(aim_list) > num_sxns:
    ctr_idx = aim_list.index(a_cent)
    if (ctr_idx+incr >= len(aim_list)-1) or (ctr_idx-incr <= 2):
          cycle_count=0 # restart at center, now we will be removing the odds if started even or vise versa
    incr=1+cycle_count
    if flip == 1:
        aim_list.pop( (ctr_idx+incr)) # remove element after center
        flip = -1
    else: 
        aim_list.pop( (ctr_idx-incr)) # remove element before center
        flip =  1
        cycle_count+=1 # only increase the cycle count after a pair has been removed
    print(aim_list)

