set cluster_size = (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 40 50 60 70 80 90 100) 
set npoints = 1120
foreach i ($cluster_size) 
 set energy_output=`paste DFT_ENERGY_WAT"$i"_"$npoints" ML_ENERGY_WAT"$i"_"$npoints" | awk '{print sqrt(($1-$2)*($1-$2));}' | awk '{sum+=$1;} END{print 27.2116*1000.0*sum/(3*'$i'*NR);}'`
 set force_output=`paste DFT_FORCES_WAT"$i"_"$npoints" ML_FORCES_WAT"$i"_"$npoints" | awk '{print sqrt(($1-$4)*($1-$4)),sqrt(($2-$5)*($2-$5)),sqrt(($3-$6)*($3-$6));}' | awk '{sumx+=$1;sumy+=$2;sumz+=$3;} END{print (27.2116*1000/0.529177)*(sumx+sumy+sumz)/(3*NR),(27.2116*1000/0.529177)*sumx/NR,(27.2116*1000/0.529177)*sumy/NR,(27.2116*1000/0.529177)*sumz/NR;}'`
 echo "WATER_"$i "ENERGY_MAE(meV/atom):" $energy_output "FORCES_MAE(meV/Angstrom): [AVE X Y Z]" $force_output
end

set cluster_size = (200 300 400 500 600 700 800 900 1000)
set npoints = 112
foreach i ($cluster_size)
 set energy_output=`paste DFT_ENERGY_WAT"$i"_"$npoints" ML_ENERGY_WAT"$i"_"$npoints" | awk '{print sqrt(($1-$2)*($1-$2));}' | awk '{sum+=$1;} END{print 27.2116*1000.0*sum/(3*'$i'*NR);}'`
 set force_output=`paste DFT_FORCES_WAT"$i"_"$npoints" ML_FORCES_WAT"$i"_"$npoints" | awk '{print sqrt(($1-$4)*($1-$4)),sqrt(($2-$5)*($2-$5)),sqrt(($3-$6)*($3-$6));}' | awk '{sumx+=$1;sumy+=$2;sumz+=$3;} END{print (27.2116*1000/0.529177)*(sumx+sumy+sumz)/(3*NR),(27.2116*1000/0.529177)*sumx/NR,(27.2116*1000/0.529177)*sumy/NR,(27.2116*1000/0.529177)*sumz/NR;}'`
 echo "WATER_"$i "ENERGY_MAE(meV/atom):" $energy_output "FORCES_MAE(meV/Angstrom): [AVE X Y Z]" $force_output
end
