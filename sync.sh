if [ -d "/cygdrive/d/" ] 
then    
    ROOT=/cygdrive/d
else
    ROOT=/cygdrive/c/Downloads
fi
    
echo "172.20.117.215"
rsync shaohua@172.20.117.215:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/117.215-vit-stufeat-sg/ -aic 2>&1|grep -E "fcstp|f\\+\\+"
echo "172.20.74.65"
rsync shaohua@172.20.74.65:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/74.65-rvgg-stufeat-sg/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
echo "10.2.18.238"
rsync shaohua@10.2.18.238:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/18.238-rvgg-sg-align/ -aic  2>&1|grep -E "fcstp|f\+\+"
echo "10.2.18.254"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/18.254-vit-sg-align/ -aic  2>&1|grep -E "fcstp|f\+\+"
