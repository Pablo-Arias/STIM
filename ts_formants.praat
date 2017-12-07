form Variables
    sentence filename
endform
Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Formant (burg)... 0.001 5 5500 0.005 50
Rename: "formant"

for i to (tmax-tmin)/0.001
	time = tmin + i * 0.001
	select Formant formant
	
	f1 = Get value at time: 1, time, "Hertz", "Linear"
	f1_bw = Get bandwidth at time: 1, time, "Hertz", "Linear"
	f2 = Get value at time: 2, time, "Hertz", "Linear"
	f2_bw = Get bandwidth at time: 2, time, "Hertz", "Linear"
	f3 = Get value at time: 3, time, "Hertz", "Linear"
	f3_bw = Get bandwidth at time: 3, time, "Hertz", "Linear"
	f4 = Get value at time: 4, time, "Hertz", "Linear"
	f4_bw = Get bandwidth at time: 4, time, "Hertz", "Linear"
	f5 = Get value at time: 5, time, "Hertz", "Linear"
	f5_bw = Get bandwidth at time: 5, time, "Hertz", "Linear"

    	appendInfoLine:  fixed$ (time, 3) ," ", fixed$ (f1, 2)," ", fixed$ (f2, 2)," ", fixed$ (f3, 2)," ", fixed$ (f4, 2)," ", fixed$ (f5, 2)," ", fixed$ (f1_bw, 2)," ", fixed$ (f2_bw, 2)," ", fixed$ (f3_bw, 2)," ", fixed$ (f4_bw, 2)," ", fixed$ (f5_bw, 2)
endfor
