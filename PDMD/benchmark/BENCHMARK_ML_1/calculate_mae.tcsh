set i = 1 
while ( $i <= 21 )
 set energy_output=`paste DFT_ENERGY_WAT"$i"_1120 ML_ENERGY_WAT"$i"_1120 | awk '{print sqrt(($1-$2)*($1-$2));}' | awk '{sum+=$1;} END{print 27.2116*1000.0*sum/(3*'$i'*NR);}'`
 set force_output=`paste DFT_FORCES_WAT"$i"_1120 ML_FORCES_WAT"$i"_1120 | awk '{print sqrt(($1-$4)*($1-$4)),sqrt(($2-$5)*($2-$5)),sqrt(($3-$6)*($3-$6));}' | awk '{sumx+=$1;sumy+=$2;sumz+=$3;} END{print (27.2116*1000/0.529177)*(sumx+sumy+sumz)/(3*NR),(27.2116*1000/0.529177)*sumx/NR,(27.2116*1000/0.529177)*sumy/NR,(27.2116*1000/0.529177)*sumz/NR;}'`
 echo "WATER_"$i "ENERGY_MAE(meV/atom):" $energy_output "FORCES_MAE(meV/Angstrom): [AVE X Y Z]" $force_output
 @ i++
end

set i = 1 
while ( $i <= 21 )
 set energy_output=`paste DFT_ENERGY_WAT"$i"_1120 ML_ENERGY_WAT"$i"_1120 | awk '{print sqrt(($1-$2)*($1-$2));}' | awk '{sum+=$1;} END{print sum/NR;}'`
 set force_output=`paste DFT_FORCES_WAT"$i"_1120 ML_FORCES_WAT"$i"_1120 | awk '{print sqrt(($1-$4)*($1-$4)),sqrt(($2-$5)*($2-$5)),sqrt(($3-$6)*($3-$6));}' | awk '{sumx+=$1;sumy+=$2;sumz+=$3;} END{print (sumx+sumy+sumz)/(3*NR),sumx/NR,sumy/NR,sumz/NR;}'`
 echo "WATER_"$i "ENERGY_MAE:" $energy_output "FORCES_MAE: [AVE X Y Z]" $force_output
 @ i++
end

