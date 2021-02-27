processing:
	@echo "Processing all participants..."
	python3 proc_0_readin_to_mne.py
	python3 proc_1_calculate_spectral_param_electrodes.py
	python3 proc_2_compute_ssd_selected_peaks.py
	python3 proc_3_calculate_bursts_ssd.py
	python3 proc_4_calculate_spectral_param_ssd.py

figures:
	@echo "Generating figures..."
	ls fig_*.py|xargs -n 1 -P 2 python3

all:
	processing figures
