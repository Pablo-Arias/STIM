form Variables
    sentence filename
	positive time_step
	positive pitch_floor
	positive max_nb_candidates
	sentence accuracy 
	positive silence_threshold
	positive voicing_threshold
	positive octave_cost
	positive octave_jump
	positive voice_unvoiced_cost
	positive pitch_ceiling
endform

Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Pitch (cc)... time_step pitch_floor max_nb_candidates accuracy silence_threshold voicing_threshold octave_cost octave_jump voice_unvoiced_cost pitch_ceiling

Rename: "pitch"

for i to (tmax-tmin)/time_step
	time = tmin + i * time_step
	
	f1 = Get value at time: time, "Hertz", "Linear"

    appendInfoLine:  fixed$ (time, 3) ," ", fixed$ (f1, 2)
endfor
