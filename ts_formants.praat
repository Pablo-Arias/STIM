form Variables
    sentence filename
	positive time_step
	positive nb_formants
	positive max_freq
	positive window_size
	positive pre_emph

endform
Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Formant (burg)... time_step nb_formants max_freq window_size pre_emph
Rename: "formant"

for i to (tmax-tmin)/time_step
	time = tmin + i * time_step
	select Formant formant

	sentence$ = fixed$ (time, 3) + " "
	for nb_formant from 1 to nb_formants
		f[nb_formant] = Get value at time: nb_formant, time, "Hertz", "Linear"
		bw[nb_formant] = Get bandwidth at time: nb_formant, time, "Hertz", "Linear"
		sentence$ = sentence$ +" " + fixed$(f[nb_formant], 2) + " " + fixed$(bw[nb_formant], 2)
	endfor
	appendInfoLine: sentence$

endfor
