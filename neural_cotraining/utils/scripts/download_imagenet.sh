cd /work/data/image_classification
mkdir ImageNet
cd ImageNet
apt-get install ctorrent
wget https://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent
ctorrent 5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent -E 1:10000
wget https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent
ctorrent a306397ccf9c2ead27155983c254227c0fd938e2.torrent -E 1:10000
mkdir -p val/images
tar xfv ILSVRC2012_img_val.tar -C val/images
cd val/images
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh
./valprep.sh
cd ../../
mkdir train
tar xfv ILSVRC2012_img_train.tar -C train
cd train
for file in n*.tar; do dir=${file%.tar} && mkdir -p $dir && tar xvf "${file}" -C $dir && rm "${file}"; done

