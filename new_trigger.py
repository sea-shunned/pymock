# if delta_val != 0:

# 	if gen == initial_gens:
# 		init_grad = (HV[-1] - HV[0]) / len(HV)


# 	elif gen > initial_gens:
# 		pass


delta_val = 90
block_trigger_gens = 10
adapt_gens = [0]
window_size = 3

# HV = list(range(0,50,4))

HV = list(range(1990,2001,1))

for gen in range(1,100):
	if delta_val != 0:
		if gen >= (adapt_gens[-1] + block_trigger_gens):

			if gen == (adapt_gens[-1] + block_trigger_gens):
				ref_grad = (HV[-1] - HV[0]) / len(HV)
				print("Here at the equals bit",gen)
				print(ref_grad)
				continue

			print("Here after first if at",gen)

			curr_grad = (HV[-1] - HV[-(window_size+1)]) / window_size
			# print(curr_grad)

			if curr_grad < 0.5 * ref_grad:
				print("Here inside the trigger at",gen)
				adapt_gens.append(gen)

				# Reset our block (to ensure it isn't the initial default)
				block_trigger_gens = 5

	HV.append(HV[-1]+0.25)

print(HV)
print(adapt_gens)

delta_val = 90
block_trigger_gens = 10
adapt_gens = [0]
window_size = 3

HV = list(range(1990,2001,1))

for gen in range(1,100):
	if gen == block_trigger_gens:
		ref_grad = (HV[-1] - HV[0]) / len(HV)

	elif gen > block_trigger_gens:
		curr_grad = (HV[-1] - HV[-(window_size+1)]) / window_size

		if gen >= adapt_gens[-1] + block_trigger_gens:
			if curr_grad < 0.5 * ref_grad:

				adapt_gens.append(gen)

				block_trigger_gens = 5

	HV.append(HV[-1]+0.25)

print(HV)
print(adapt_gens)