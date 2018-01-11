	colours = [Color('steelblue'), Color('orange'), Color('green'), Color('red'), Color('purple'), Color('maroon'), Color('navy'), Color('gray'), Color('olive'), Color('deepskyblue')]
	

	ind = np.arange(int(n_out/2)*-2, int(n_out/2))
	ind = ind*0.36

	w=0.16

	ens_prob_avg = {}
	ens_spike_avg = {}

	for c in range(n_out):
		avg = np.zeros(n_out)
		avg_spike = np.zeros(n_out)
		for e in range(int(noExPerClass[c])):
			avg += ens_prob[c][e]
			avg_spike += ens_spike_prob[c][e]
		ens_prob_avg[c] = avg/int(noExPerClass[c])
		ens_spike_avg[c] = avg_spike/int(noExPerClass[c])


	labels_acc_mfr = []
	for j in range(n_out):
		labels_acc_mfr.append('o'+str(j))

	fig, (ax1, ax2) = plt.subplots(2, figsize=(12,12))
	for i in range(int(n_out/2)):
		for j in range(n_out):
			# labels_acc_mfr.append('o'+str(j))
			g1=ax1.bar(4*i+ind[j], ens_prob_avg[i][j], width=w, alpha=0.6 , color=colours[0].rgb, label='geometric comb')
			n1=ax1.bar(4*i+ind[j]+0.18, ens_spike_avg[i][j], width=w, alpha=0.6 , color=colours[1].rgb, label='neuron comb')

			ax1.text(4*i+ind[j], ens_prob_avg[i][j], labels_acc_mfr[j], ha='center', va='bottom', color=Color('darkslategray').rgb, rotation='vertical')
			ax1.text(4*i+ind[j]+0.18, ens_spike_avg[i][j], labels_acc_mfr[j], ha='center', va='bottom', color=Color('darkslategray').rgb, rotation='vertical')

# secnd subplot showing the remaining classes
	for i in range(int(n_out/2), n_out):
		for j in range(n_out):
			g2=ax2.bar(4*i+ind[j], ens_prob_avg[i][j], width=w, alpha=0.6 , color=colours[0].rgb, label='geometric comb')
			n2=ax2.bar(4*i+ind[j]+0.18, ens_spike_avg[i][j], width=w, alpha=0.6 , color=colours[1].rgb, label='neuron comb')

			ax2.text(4*i+ind[j], ens_prob_avg[i][j], labels_acc_mfr[j], ha='center', va='bottom', color=Color('darkslategray').rgb, rotation='vertical')
			ax2.text(4*i+ind[j]+0.18, ens_spike_avg[i][j], labels_acc_mfr[j], ha='center', va='bottom', color=Color('darkslategray').rgb, rotation='vertical')

	ax1.set_xticks(np.arange(5)*4+ind[5])
	ax1.set_xticklabels(('Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'))
	ax1.set_xlabel('True class label')
	ax1.set_ylabel('Probability')
	ax1.legend(handles=[g1,n1])
	ax1.set_title('Comparison of the probablities assigned by the geometric combiner and the neuron combiner for each true class.')


	ax2.set_xticks(np.arange(5,10)*4+ind[5])
	ax2.set_xticklabels(('Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'))
	ax2.set_xlabel('True class label')
	ax2.legend(handles=[g2,n2])
	ax2.set_ylabel('Probability')

	plt.tight_layout()
	plt.show()