### Requirements:
* Sel-CL:
** Python 3.8.10
** Pytorch 1.8.0 (torchvision 0.9.0)
** Numpy 1.19.5
** scikit-learn 1.0.1
** apex 0.1

* Unicon:
** numpy
** opencv-python
** Pillow
** tensorboardX
** torch
** torchnet
** torchvision
** tqdm

### Operation:
* Sel-CL:
** Train: 
python3 train_Sel-CL-New.py --epoch=200 --num_classes=10 --batch_size=128 --low_dim=128 --lr-scheduler="step" --noise_ratio=0.2 --network="PR18" --lr=0.1 --wd=1e-4 --dataset="CIFAR-10" --download=True --noise_type="symmetric"  --sup_t=0.1 --headType="Linear"  --sup_queue_use=1 --sup_queue_begin=3 --queue_per_class=1000  --alpha=0.5 --beta=0.25 --k_val=250 --poisoning_rate=0.2 --trigger_label=0

** Test:
python3 test_acc.py --dataset CIFAR-10 --load_local

* Unicon:
** Train and Test:
Python3 Train_cifar.py



