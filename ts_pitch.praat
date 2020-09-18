form Variables
    sentence filename
	positive time_step
	positive pitch_floor
	positive pitch_ceiling
endform

Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Pitch... time_step pitch_floor pitch_ceiling
Rename: "pitch"

for i to (tmax-tmin)/time_step
	time = tmin + i * time_step
	
	f1 = Get value at time: time, "Hertz", "Linear"

    appendInfoLine:  fixed$ (time, 3) ," ", fixed$ (f1, 2)
endfor
