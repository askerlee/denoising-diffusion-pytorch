if [ -d "/cygdrive/d/" ] 
then    
    ROOT=/cygdrive/d
else
    ROOT=/cygdrive/c/Downloads
fi
    
echo "172.20.74.65"
rsync shaohua@172.20.74.65:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/74.65-clsinterp2-tea-geo-bs96/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
#echo "172.20.117.215"
#rsync shaohua@172.20.117.215:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/117.215-clsinterp-tea-geo-bs128/ -aic 2>&1|grep -E "fcstp|f\\+\\+"
#echo "10.2.18.238"
#rsync shaohua@10.2.18.238:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/18.238-clsguide-repvgg-full/ -aic  2>&1|grep -E "fcstp|f\+\+"
echo "10.2.18.254-1"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/18.254-imagenet-clssinterp-tea-bs96/ -aic  2>&1|grep -E "fcstp|f\+\+"
#echo "10.2.18.254-2"
#rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results2/'*png' $ROOT/denoising/18.254-dualcls-tea-repvgg/ -aic  2>&1|grep -E "fcstp|f\+\+"
