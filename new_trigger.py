if delta_val != 0:

	if gen == initial_gens:
		init_grad = (HV[-1] - HV[0]) / len(HV)


	elif gen > initial_gens:
		pass



block_trigger_gens = 10
adapt_gens = [0]

HV.append(hypervolume(pop, HV_ref))

if delta_val != 0:
	if gen >= (adapt_gens[-1] + block_trigger_gens):

		if gen == (adapt_gens[-1] + block_trigger_gens):
			ref_grad = (HV[-1] - HV[0]) / len(HV)
			print("Here at the equals bit",gen)

		print("Here after first if at",gen)

		curr_grad = (curr_HV - HV[-(window_size+1)]) / window_size

		if curr_grad < 0.5 * ref_grad:
			print("Here inside the trigger at",gen)
			adapt_gens.append(gen)

			# Reset our block (to ensure it isn't the initial default)
			block_trigger_gens = 5

