# SymNet_torch_dev
SymNet_torch development repo

# requirement

See `requirementes.txt`. Python 3.7 + PyTorch 1.8.1

# Download data
Run in data
    bash download_data.sh 
# usage


MIT

    python run_symnet.py --network fc_obj --name MIT_obj_lr3e-3 --data MIT --epoch 1500 --batchnorm --lr 3e-3
    python test_obj.py --network fc_obj --name MIT_obj_lr3e-3 --data MIT --epoch 1120 --batchnorm
    python run_symnet.py --name MIT_best --data MIT --epoch 400 --obj_pred MIT_obj_lr3e-3_ep1120.pt --batchnorm --lr 5e-4 --bz 512 --lambda_cls_attr 1 --lambda_cls_obj 0.01 --lambda_trip 0.03 --lambda_sym 0.05 --lambda_axiom 0.01
    python test_symnet.py --name MIT_best --data MIT --epoch 320 --obj_pred MIT_obj_lr3e-3_ep1120.pt --batchnorm

UT

    python run_symnet.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 300 --batchnorm --lr 1e-3
    python test_obj.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 140 --batchnorm
    python run_symnet.py --name UT_best --data UT --epoch 700 --obj_pred UT_obj_lr1e-3_ep140.pt --batchnorm  --wordvec onehot  --lr 1e-4 --bz 256 --lambda_cls_attr 1 --lambda_cls_obj 0.5 --lambda_trip 0.5 --lambda_sym 0.01 --lambda_axiom 0.03
    python test_symnet.py --name UT_best --data UT --epoch 600 --obj_pred UT_obj_lr1e-3_ep140.pt --wordvec onehot --batchnorm


# 如果有文件夹已存在问题,加--force参数后直接跑！
其他参数只要object classifier用的版本队的就没大问题，不用改，如果觉得训的太慢调epoch。
# TODOs

1. yaml以及自动备份
0. MSEloss在L2的时候不对：少个平方
1. activation function和weight initializer没有设置
3. args的key名字跟operator不太一样，可以考虑统一一下
7. lr scheduler还没实现。如果要加的话还要存进statedict
8. GRADIENT_CLIPPING还没实现
9. focal loss not implemented
10. loss的log精简一下，tb不要显示那么多（参考tf版本
11. reshape->view
13. make this repo more Python3 (type, etc.)
14. 检查从snapshot继续训练时读取有没有错，分数是不是合理