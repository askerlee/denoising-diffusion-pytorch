if [ -d "/cygdrive/d/" ] 
then    
    ROOT=/cygdrive/d
else
    ROOT=/cygdrive/c/Downloads
fi
    
echo "172.20.74.65"
rsync -r shaohua@172.20.74.65:denoising-diffusion-pytorch/samples/ $ROOT/denoising/74.65-pok-interpclip100-tea/ -aic  2>&1|grep -E "fcstp|f\\+\\+"
#echo "172.20.117.215"
#rsync -r shaohua@172.20.117.215:denoising-diffusion-pytorch/samples/ $ROOT/denoising/117.215-pok-interpvitclamp/ -aic 2>&1|grep -E "fcstp|f\\+\\+"
#echo "10.2.18.238"
#rsync shaohua@10.2.18.238:denoising-diffusion-pytorch/results/'*png' $ROOT/denoising/18.238-clsguide-repvgg-full/ -aic  2>&1|grep -E "fcstp|f\+\+"
#echo "10.2.18.254-1"
#rsync -r li_shaohua@10.2.18.254:denoising-diffusion-pytorch/samples/ $ROOT/denoising/18.254-clssingle0.01-tea-bs96/ -aic  2>&1|grep -E "fcstp|f\+\+"
#echo "A100"
#rsync -r shaohua@172.20.117.175:denoising-diffusion-pytorch/samples/ $ROOT/denoising/A100-flower-clssinterp-tea-bs64/ -aic  2>&1|grep -E "fcstp|f\+\+"
#echo "A100-1"
#rsync -r shaohua@172.20.117.175:denoising-diffusion-pytorch/samples/ $ROOT/denoising/A100-flower-interpvitclamp-tea/ -aic  2>&1|grep -E "fcstp|f\+\+"
echo "A100-2"
rsync -r shaohua@172.20.117.175:denoising-diffusion-pytorch/samples2/ $ROOT/denoising/A100-flower-interpvit-tea/ -aic  2>&1|grep -E "fcstp|f\+\+"
