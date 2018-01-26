'''

r-ticul8 0.0

Web Application for Insight Health Data Science Fellowship, Boston, MA -- 2018A

Author: Abdullah Al Rashid

Last Revision: 2018.01.26


'''


import wave as wv
import os 
import struct as st
import numpy as np
import pickle
import matplotlib.pyplot as mplt
from sklearn.externals import joblib

wav_file_samp_freq = 1.6e4
wav_file_dur = 0.4
delta_f = 1.0 / wav_file_dur
pickle_name_ref_dict = "PowerDistrRef.pkl"
joblib_name_rf_clf = "RandomForests.jlb"
pickle_name_app_feat_dict = "ImportantAppFeatures_UnimpairedDataset.pkl"


# Parsing Power Spectrum (Integral Normalised from a Single .wav File
def power_spect_single(file_path, file_name, sampling_freq, max_duration):
	p = [] # Placeholder for power spectrum
	
	# Basic Temporal and Frequency Data
	len_t = 1 + int(max_duration*sampling_freq)
	len_f = 1 + int(sampling_freq*max_duration)
	df = 1.0 / max_duration
		
	if (file_name[-4:] == ".wav") and os.path.exists(os.path.join(file_path, file_name)):
		# Reading .wav File
		wav_file = wv.open(os.path.join(file_path, file_name), 'r')
		wav_frames = wav_file.readframes(-1)
		sample_width = wav_file.getsampwidth()
		s_on_file = st.unpack('h'*wav_file.getnframes(), wav_frames)
		frame_rate = wav_file.getframerate()
		len_t_on_file = len(s_on_file)
		wav_file.close()

		s = np.zeros(len_t)
		s[:min(len_t_on_file, len_t)] = s_on_file[:min(len_t_on_file, len_t)]	

		s_fft = np.fft.fft(s)         
		s_fft = s_fft[:len(s_fft)/2] # Removal of negative-frequency information  

		p_on_file = np.abs(s_fft)
		p = np.zeros(len_f)
		p[:min(len_f, len(s_fft))] = p_on_file[:min(len_f, len(s_fft))]
		norm_factor_p = 1.0 / (sum(p) * df)
		p *= norm_factor_p

	return p

# Calculation of Hellinger Proximity (Cf. 'Hellinger Distance' on Wikipedia) of Two Power Spectra	
def hellinger_proximity(p, q, del_freqs):
	return sum(np.sqrt(p*q))*del_freqs

# Major Processing of a Sample Dataset Uploaded by User
# Outputs: 
#	1. Formatted test data for Random Forest Classifier
#	2. 'dict' object of background data for top 20 features used in app
#	3. 'dict' object of data for top 20 features for current sample
def process_sample():
	pwr_dict_ref = pickle.load(open(os.path.join(os.path.join(os.getcwd(), "pickles"), pickle_name_ref_dict), "rb"))
	wav_file_key_list = sorted(pwr_dict_ref.keys())

#	print "# Diagnostic [process_sample(...)]: wav_file_key_list = " + str(wav_file_key_list)

	app_feature_data_dict =	pickle.load(open(os.path.join(os.path.join(os.getcwd(), "pickles"), pickle_name_app_feat_dict), "rb"))
#	app_feature_labels =  pickle.load(open(os.path.join(os.path.join(os.getcwd(), "pickles"), pickle_name_app_feat_labels), "rb"))
	sample_feat_data_dict = {}

	x_samp = []
	X_test = []
	
	for idx0 in range(len(wav_file_key_list)):
		key_ref = wav_file_key_list[idx0]
		if key_ref == '':
			continue
		for idx1 in range(len(wav_file_key_list)):
			key_samp = wav_file_key_list[idx1]
			if key_samp == '':
				continue
			samp_file_name = key_samp + ".wav"
			feature_key = key_ref + "__" + key_samp
			feature_value = -1.0
			if os.path.exists(os.path.join(os.path.join(os.getcwd(), "uploads"), samp_file_name)):
				pwr_dist_samp = power_spect_single(os.path.join(os.getcwd(), "uploads"), samp_file_name, wav_file_samp_freq, wav_file_dur)
				feature_value = hellinger_proximity(pwr_dict_ref[key_ref], pwr_dist_samp, delta_f)
			x_samp.append(feature_value)

			if (feature_key in app_feature_data_dict) and (feature_value != -1.0):
				sample_feat_data_dict[feature_key] = [feature_value, bool( 1.0 < (abs(feature_value - app_feature_data_dict[feature_key][0]) / app_feature_data_dict[feature_key][1]) )]
				
	X_test.append(x_samp)

	return np.array(X_test, dtype=float), app_feature_data_dict, sample_feat_data_dict

# Prediction of Formatted Input via a Random Forest Classifier
def sample_prediction(X_test):
#	print "# Diagnostic [sample_prediction(...)]: X_test.shape = " + str(X_test.shape)
	rf_clf = joblib.load(open(os.path.join(os.path.join(os.getcwd(), "pickles"), joblib_name_rf_clf), "rb"))
	return rf_clf.predict(X_test)

# Ancillairy Function -- List Reversal 
def reverse_list(items):
	return [items[len(items) - 1 - idx] for idx in range(len(items))]

# Major Step for Result/Output-Generation
def generate_results(app_feature_data_dict, sample_feat_data_dict, predicted_class, image_token=0):
	all_app_features = app_feature_data_dict.keys()
	num_features_app = len(all_app_features)
	available_app_features = sample_feat_data_dict.keys()

	flags = []
	flagged_phonemes_dict = {}

	#bp_fig = mplt.figure(figsize=(16, num_features_app*0.8))
	#bp_ax = bp_fig.add_subplot(1,1,1)

	# MatPlotLib Figure Containing Sub-Plots for Range and Box Plots
	rslt_fig, (rng_ax, bp_ax) = mplt.subplots(2, figsize=(16, num_features_app*1.6), sharey=True)

	app_feature_labels = []	# Labels for box plots
	for feature in all_app_features:
		ref_phoneme, samp_phoneme = feature.split("__")		
		app_feature_labels.append('\'' + samp_phoneme + "\'\n(vs. \'" + ref_phoneme + "\' refc.)")
	
	box_properties = dict(linestyle='-', linewidth=3, color='c', alpha=0.3)
	median_properties = dict(linestyle='-', linewidth=2.5, color='y')
	meanline_properties = dict(linestyle='-', linewidth=2.5, color="purple")
	feature_text_properties_neu = dict(color='b', fontsize=16)
	feature_text_properties_pos = dict(color='g', fontsize=16)
	feature_text_properties_neg = dict(color='r', fontsize=16)
	
	plot_boxes = []
	y_feature = 0.0
	for feature in app_feature_data_dict:
		y_feature += 1.0
		plot_boxes.append(app_feature_data_dict[feature][2])
		
		x_feature_text = 0.05
		
		# Range Plot Background Data
		rng_ax.scatter(app_feature_data_dict[feature][0], y_feature, marker='|', color='k', s=400, alpha=0.3)
		rng_ax.scatter(app_feature_data_dict[feature][0] - app_feature_data_dict[feature][1], y_feature, marker='>', color='k', s=400, alpha=0.3)
		rng_ax.scatter(app_feature_data_dict[feature][0] + app_feature_data_dict[feature][1], y_feature, marker='<', color='k', s=400, alpha=0.3)
		 
		# Data Plotting with Phoneme Pair Flags for Further Investigation
		if predicted_class == 0:	# Unimpaired subject
			if feature in sample_feat_data_dict:
				bp_ax.scatter([sample_feat_data_dict[feature][0]], [y_feature], color='b', marker='o', s=200)
				rng_ax.scatter([sample_feat_data_dict[feature][0]], [y_feature], color='b', marker='o', s=200)
				bp_ax.text(x_feature_text, y_feature, "NO_FLAG", fontdict=feature_text_properties_neu)
				rng_ax.text(x_feature_text, y_feature, "NO_FLAG", fontdict=feature_text_properties_neu)
			else:
				bp_ax.text(x_feature_text, y_feature, "UNAVAILABLE", fontdict=feature_text_properties_neu)
				rng_ax.text(x_feature_text, y_feature, "UNAVAILABLE", fontdict=feature_text_properties_neu)
		else:	# Impaired subject
			if feature in sample_feat_data_dict:	# Feature sought for app contained in sample
				data_colour = 'g'
				feature_flag = sample_feat_data_dict[feature][1]
	
				if feature_flag:	# Flagged feature
					data_colour = 'r'
					bp_ax.text(x_feature_text, y_feature, "FLAGGED", fontdict=feature_text_properties_neg)
					rng_ax.text(x_feature_text, y_feature, "FLAGGED", fontdict=feature_text_properties_neg)
					ref_phoneme, samp_phoneme = feature.split("__") 
					if (samp_phoneme in flagged_phonemes_dict):
						flagged_phonemes_dict[samp_phoneme].append(ref_phoneme)
					else:
						flagged_phonemes_dict[samp_phoneme] = [ref_phoneme]
				else:	# Un-flagged feature
					bp_ax.text(x_feature_text, y_feature, "NO_FLAG", fontdict=feature_text_properties_pos)	
					rng_ax.text(x_feature_text, y_feature, "NO_FLAG", fontdict=feature_text_properties_pos)
				
				bp_ax.scatter([sample_feat_data_dict[feature][0]], [y_feature], color=data_colour, marker='o', s=200)
				rng_ax.scatter([sample_feat_data_dict[feature][0]], [y_feature], color=data_colour, marker='o', s=200)
			else:	# Feature sought not in sample
				bp_ax.text(x_feature_text, y_feature, "UNAVAILABLE", fontdict=feature_text_properties_neu)
				rng_ax.text(x_feature_text, y_feature, "UNAVAILABLE", fontdict=feature_text_properties_neu)
			
	bp_ax.boxplot(reverse_list(plot_boxes), 0, 'bs', 0, showfliers=False, boxprops=box_properties, meanprops=meanline_properties, meanline=True, medianprops=median_properties)	# Use of reverse_list(...) to plot most important feature at the top
	

	mplt.xticks(np.arange(0.0, 1.01, 0.2), np.arange(0.0, 1.01, 0.2))
	mplt.yticks(np.arange(1, num_features_app + 1.1, 1), reverse_list(app_feature_labels), rotation=0)	# Use of reverse_list(...) to plot most important feature at the top
	bp_ax.grid()
	rng_ax.grid()
	bp_ax.set_title("Box Plots: Ordinality Information", fontsize=16)
	rng_ax.set_title("Mean $\pm$ 1x Standard Deviation Range Plots: Normalcy Information", fontsize=16)
	bp_ax.set_xlim([0.0, 1.0])
	rng_ax.set_xlim([0.0, 1.0])
	
	# Unique Image File Name Using a Random Number Token to Circumvent Browser Caching Issue While Serving Results Live
	if image_token==0:
		rslt_fig.savefig(os.path.join(os.path.join(os.path.join(os.getcwd(), "images"), "Results.png")))
	else:
		rslt_fig.savefig(os.path.join(os.path.join(os.path.join(os.getcwd(), "images"), "Results_" + str(image_token) + ".png")))
	
	# Formatting Text of Phoneme Suggestions for Further Investigation
	if len(flagged_phonemes_dict) == 0:
		flags.append("None")
	else:
		flags.append("Phoneme(s) listed below are suggested for further testing:")
		for samp_key in flagged_phonemes_dict:
			samp_phoneme_flag_str = "* \'" + samp_key + "\' with respect to reference(s)" 
			prefix_char = ''
			for ref_key in flagged_phonemes_dict[samp_key]:
				samp_phoneme_flag_str += prefix_char + ' ' + '\'' + ref_key + '\''
				if prefix_char == '':
					prefix_char = ','
			
			flags.append(samp_phoneme_flag_str)
		flags.append("Note: Testing is recommended for sample and reference phoneme similarity value lying more than one standard deviation from the mean values for unimpaired speech data.")
		flags.append("Disclaimer: Phoneme tesing suggestion(s) do not consitute confirmatory diagnosis. Further testing must be undertaken.")
	
	return flags

# Deletion of Uploaded Files
def delete_uploaded_files():
	for file_name in os.listdir(os.path.join(os.getcwd(), "uploads")):
    		file_path = os.path.join(os.path.join(os.getcwd(), "uploads"), file_name)
    		try:
        		if os.path.isfile(file_path):
            			os.unlink(file_path)
        		#elif os.path.isdir(file_path): shutil.rmtree(file_path)
    		except Exception as e:
        		print(e)

# Deletion of Image Results
def delete_result_image():
	results_file_name = ''
	for file_name in os.listdir(os.path.join(os.getcwd(), "images")):
		if ("Results" in file_name):
			results_file_name = file_name
			break		
	
	file_path = os.path.join(os.path.join(os.getcwd(), "images"), results_file_name)
	if os.path.exists(file_path):
		try:
			if os.path.isfile(file_path):
	    			os.unlink(file_path)
		except Exception as e:
			print(e)

