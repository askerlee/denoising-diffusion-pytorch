if [ -d "/cygdrive/d/" ] 
then    
    ROOT=/cygdrive/d
else
    ROOT=/cygdrive/c/Downloads
fi
    
echo "172.20.117.215"
rsync shaohua@172.20.117.215:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/117.215-vit-noise0.25-frz-align0.01/ -aic 2>&1|grep -E "fcstp|f\\+\\+"
echo "172.20.74.65"
rsync shaohua@172.20.74.65:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/74.65-vit-noise0.85-tune-align0.1/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
echo "10.2.18.238"
rsync shaohua@10.2.18.238:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/18.238-vit-noise0.85-tune-align0.001/ -aic  2>&1|grep -E "fcstp|f\+\+"
echo "10.2.18.254"
rsync li_shaohua@10.2.18.254:denoising-diffusion-pytorch/results/'sample*png' $ROOT/denoising/18.254-vit-noise0.85-tune-align0.01/ -aic  2>&1|grep -E "fcstp|f\+\+"
