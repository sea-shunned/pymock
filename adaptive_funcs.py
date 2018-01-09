import numpy as np

def triggerGens_random(num_gens):
	upper = int(num_gens-(num_gens/10))
	lower = int(num_gens/10)

	trigger_gens = sorted([np.random.randint(lower,upper) for i in range(0,4)])

	for i in range(len(trigger_gens)):
		try:
			if trigger_gens[i] + lower >= trigger_gens[i+1]:
				trigger_gens[i+1] = trigger_gens[i] + lower

		except IndexError:
			continue

	if np.max(trigger_gens) >= 100:
		trigger_gens = [i-10 for i in trigger_gens]

	# return sorted([np.random.randint(lower,upper) for i in range(0,4)])
	return trigger_gens

def triggerGens_interval(num_gens):
	upper = int(num_gens-(num_gens/10))
	lower = int(num_gens/10)
	interval = (upper-lower)/4

	trigger_gens = [np.random.randint(lower+(interval*i), lower+(interval*(i+1))) for i in range(0,4)]

	return trigger_gens

if __name__ == '__main__':
	# for i in range(100):
		# print(triggerGens_random(100))
		# print(triggerGens_interval(100),"\n")
		# print(gens)

	print([triggerGens_interval(100) for i in range(30)])

		# if 70 in gens:
		# 	print("OK")
		# else:
		# 	print("No")