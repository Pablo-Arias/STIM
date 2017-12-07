form Variables
    sentence filename
endform

Read from file... 'filename$'

To Formant (burg)... 0.001 5 5500 0.025 50
f1 = Get mean... 1 0.0 0.0 Hertz
f1$ = fixed$(f1, 2)

f2 = Get mean... 2 0.0 0.0 Hertz
f2$ = fixed$(f2, 2)

f3 = Get mean... 3 0.0 0.0 Hertz
f3$ = fixed$(f3, 2)

output$ = "Means"+newline$
output$ = output$ + f1$+ newline$
output$ = output$ + f2$+ newline$
output$ = output$ + f3$+ newline$

echo 'output$'