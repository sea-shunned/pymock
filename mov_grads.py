import numpy as np

window_size = 6


# HV = list(range(0,15,2))
# print(HV)

# curr_HV = 16
# HV.append(curr_HV)

# curr_grad = (curr_HV - HV[-(window_size+1)]) / window_size

# print(curr_grad)

# HV.append(curr_HV)


### Start again below and create grads properly
grads = []
HV = []
curr_HV = 0
for i in range(0,50,2):
	curr_HV += i
	HV.append(curr_HV)

	if len(HV) > window_size:
		curr_grad = (curr_HV - HV[-(window_size+1)]) / window_size	

		grads.append(curr_grad)
	
print(HV)
print(grads)