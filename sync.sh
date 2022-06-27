if [ -d "/cygdrive/d/" ] 
then    
    ROOT=/cygdrive/d
else
    ROOT=/cygdrive/c/Downloads
fi
    
echo "172.20.74.65"
rsync -r shaohua@172.20.74.65:denoising-diffusion-pytorch/samples/ $ROOT/denoising/74.65-pok-clsinterp-tea-bs96/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
rsync shaohua@172.20.74.65:denoising-diffusion-pytorch/results/interp/'*png' $ROOT/denoising/74.65-clssingle-tea-geo-bs96/interp/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
rsync shaohua@172.20.74.65:denoising-diffusion-pytorch/results/single/'*png' $ROOT/denoising/74.65-clssingle-tea-geo-bs96/single/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
echo "172.20.117.215"
rsync shaohua@172.20.117.215:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/117.215-pok-clssingle-tea/ -aic 2>&1|grep -E "fcstp|f\\+\\+"
#echo "10.2.18.238"
#rsync shaohua@10.2.18.238:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/18.238-clsguide-repvgg-full/ -aic  2>&1|grep -E "fcstp|f\+\+"
echo "10.2.18.254-1"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/18.254-clssingle-tea-bs96/ -aic  2>&1|grep -E "fcstp|f\+\+"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/interp/'*png' $ROOT/denoising/18.254-clssingle-tea-bs96/interp/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/single/'*png' $ROOT/denoising/18.254-clssingle-tea-bs96/single/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
#echo "10.2.18.254-2"
#rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results2/'*png' $ROOT/denoising/18.254-dualcls-tea-repvgg/ -aic  2>&1|grep -E "fcstp|f\+\+"
