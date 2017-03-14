# Loss-aware-Binaryzation
Implementation of paper "Loss aware Binarization of Deep Networks"


This repository is divided in two subrepositories:

FNN: enables the reproduction of the FNN results(on MNIST, CIFAR, SVHN)reported in the article
RNN: enables the reproduction of the RNN results(on War and Peace) reported in the article

Requirements
This software is implemented on top of the implementation of [BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect) and has all the same requirements. 


Example training command on War and Peace dataset:
- training LAB
```sh
python warpeace.py --method="LAB" --lr_start=0.002 --w="w" --len=100
```
- training LAB2
```sh
python warpeace.py --method="LAB" --lr_start=0.002 --w="wa" --len=100
```
