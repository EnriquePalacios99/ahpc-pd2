set term png

#set parametric
#set dummy t
#set autoscale
#set samples 160
#set title "Ley de Amdahl"

#set key box
set key right
set xlabel "p (#procesos)"
set ylabel "T_{p} (sec)"
set xrange[1:17]
set yrange[0.01:50]
set logscale y
#set grid ytics mytics
#set grid xtics mxtics

set title "Tiempo de Ejecucion"
set output 'times_p128.png'
plot 'times_N128.txt' u 1:2 w l lw 2 title "T_{ej}",\
'times_N128.txt' u 1:($3+$5) w l lw 2 title "T_{comm}",\
'times_N128.txt' u 1:4 w l lw 2 title "T_{comp}"
#1*(1/x+.001*log(x)+.001*x) title "teoria: 1/p+log(p)+ p" lw 3 dt 4

set output 'times_p008.png'
plot 'times_N008.txt' u 1:2 w l lw 2 title "T_{ej}",\
'times_N008.txt' u 1:($3+$5) w l lw 2 title "T_{comm}",\
'times_N008.txt' u 1:4 w l lw 2 title "T_{comp}"

