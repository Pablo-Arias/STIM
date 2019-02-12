form Variables
    sentence filename
endform
Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Pitch... 0 75.0 350.0
Rename: "pitch"

for i to (tmax-tmin)/0.001
	time = tmin + i * 0.001
	
	f1 = Get value at time: time, "Hertz", "Linear"

    appendInfoLine:  fixed$ (time, 3) ," ", fixed$ (f1, 2)
endfor
