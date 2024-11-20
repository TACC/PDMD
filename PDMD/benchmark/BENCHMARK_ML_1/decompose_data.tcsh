set iatom = 18 

paste DFT_ENERGY_WAT"$iatom"_1120 ML_ENERGY_WAT"$iatom"_1120 |  awk '{print sqrt(($1-$2)*($1-$2));}' | awk '{print $1*27.2116*1000.0/63;}' > energy_tmp
paste DFT_FORCES_WAT"$iatom"_1120 ML_FORCES_WAT"$iatom"_1120 | awk '{print sqrt(($1-$4)*($1-$4)),sqrt(($2-$5)*($2-$5)),sqrt(($3-$6)*($3-$6));}' | awk '{print ($1+$2+$3)*(27.2116*1000/0.529177)/3.0;}' > forces_tmp

set i = 1

while ( $i <= 10 )

 @ nstep = $i * 1000
 @ sline = ( $i - 1 ) * $iatom * 3 * 112 
 @ eline =  $i  * $iatom * 3 * 112 
awk '{ if ((NR>='$sline') && (NR<='$eline')) {print $1;}}' forces_tmp | awk '{sum+=$1;} END{printf("%6d %12.6f (meV/angstrom)",'$nstep',sum/NR);}'
 @ gline = ( $i - 1 ) * 112 
 @ hline =  $i  * 112 
awk '{ if ((NR>='$gline') && (NR<='$hline')) {print $1;}}' energy_tmp | awk '{sum+=$1;} END{printf("%12.6f (meV/atom)\n", sum/NR);}'

 @ i++
end
