#!/usr/bin/python
import numpy as np

#cookie price
sugar = 2.65
choco = 3.20
snick = 3.45
smore = 3.70
max_money = 10

#print type of function
print(type(max_money))

#number of cookies able to buy
#cookie first letter(s) and num
sunum = max_money // sugar
cnum = max_money // choco
snnum = max_money // snick
smnum = max_money // smore
print(sunum, cnum, snnum, smnum)

#max_money % cookie price
#cookie first letter pri
su_pri = max_money % sugar
ch_pri = max_money % choco
sn_pri = max_money % snick
sm_pri = max_money % smore
print(su_pri, ch_pri, sn_pri, sm_pri)

#reduce floats
fsu_pri = np.floor(max_money/sugar)
fch_pri = np.floor(max_money/choco)
fsn_pri = np.floor(max_money/snick)
fsm_pri = np.floor(max_money/smore)
print(fsu_pri, fch_pri, fsn_pri, fsm_pri)

#this is how far I was able to get