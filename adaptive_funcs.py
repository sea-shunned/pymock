import numpy as np

def triggerGens(num_gens):
	upper = int(num_gens-(num_gens/10))
	lower = int(num_gens/10)

	trigger_gens = sorted([np.random.randint(lower,upper) for i in range(0,4)])
	# print(trigger_gens)

	for i in range(len(trigger_gens)):
		try:
			if trigger_gens[i] + lower >= trigger_gens[i+1]:
				trigger_gens[i+1] = trigger_gens[i] + lower

		except IndexError:
			continue

	return trigger_gens



if __name__ == '__main__':
	for i in range(5):
		print(triggerGens(100))