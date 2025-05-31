set term png

#set parametric
#set dummy t
#set autoscale
#set samples 160
#set title "Ley de Amdahl"

#set key box
set key right
set xlabel "p (#procesos)"
set ylabel "T_{ej} (sec)"
set xrange[1:17]
set yrange[0.01:50]
set logscale y
#set grid ytics mytics
#set grid xtics mxtics

set title "Tiempo de Ejecucion"
set output 'times.png'
plot 'times_N001.txt' u 1:2 w l lw 2 title "N_{001}",\
'times_N002.txt' u 1:2 w l lw 2 title "N_{002}",\
'times_N004.txt' u 1:2 w l lw 2 title "N_{004}",\
'times_N008.txt' u 1:2 w l lw 2 title "N_{008}",\
'times_N016.txt' u 1:2 w l lw 2 title "N_{016}",\
'times_N032.txt' u 1:2 w l lw 2 title "N_{032}",\
'times_N064.txt' u 1:2 w l lw 2 title "N_{064}",\
'times_N128.txt' u 1:2 w l lw 2 title "N_{128}",\
1*(1/x+.001*log(x)+.001*x) title "teoria: 1/p+log(p)+ p" lw 3 dt 4

set output 'speedup.png'
set xrange[1:17]
set yrange[1:20]
unset log x
unset log y
set key left 
set title "Speedup"
set ylabel "S"
plot 'times_N001.txt' u 1:(0.1821/$2) w l lw 2 title "N_{001}",\
'times_N002.txt' u 1:(0.3775/$2) w l  lw 2 title "N_{002}",\
'times_N004.txt' u 1:(0.7811/$2) w l  lw 2 title "N_{004}",\
'times_N008.txt' u 1:(1.6211/$2) w l  lw 2 title "N_{008}",\
'times_N016.txt' u 1:(3.4438/$2) w l  lw 2 title "N_{016}",\
'times_N032.txt' u 1:(7.6623/$2) w l  lw 2 title "N_{032}",\
'times_N064.txt' u 1:(18.1341/$2) w l  lw 2 title "N_{064}",\
'times_N128.txt' u 1:(37.3285/$2) w l  lw 2 title "N_{128}",\
x title "teoria: S=O(p)" lw 2,\
1/(1/x+.001*log(x)+.001*x) title "teoria: 1/(1/p+log(p)+p)" lw 3 dt 4

set output 'eficiency.png'
set xrange[1:17]
set yrange[0:1.5]
set key left bottom 
set title "Eficiencia"
set ylabel "E"
plot 'times_N001.txt' u 1:(0.1821/$2/$1) w l lw 2 title "N_{001}",\
'times_N002.txt' u 1:(0.3775/$2/$1) w l lw 2 title "N_{002}",\
'times_N004.txt' u 1:(0.7811/$2/$1) w l lw 2 title "N_{004}",\
'times_N008.txt' u 1:(1.6211/$2/$1) w l lw 2 title "N_{008}",\
'times_N016.txt' u 1:(3.4438/$2/$1) w l lw 2 title "N_{016}",\
'times_N032.txt' u 1:(7.6623/$2/$1) w l lw 2 title "N_{032}",\
'times_N064.txt' u 1:(18.1341/$2/$1) w l lw 2 title "N_{064}",\
'times_N128.txt' u 1:(37.3285/$2/$1) w l lw 2 title "N_{128}",\
1 title "teoria: E=O(1)" lw 2,\
1/(1+.001*x*log(x)+.001*x*x) title "teoria: 1/(p*sqrt(p)+sqrt(p))" lw 3 dt 4
# (2/(1+log(x))) title "teoria: 1/(log(p))" lw 3 dt 4


