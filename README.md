# Loss-Aware-Binarization
Implementation of ICLR 2017 paper "Loss aware Binarization of Deep Networks", tested with GTX TITAN X, python 2.7, theano 0.9.0 and  lasagne 0.2.dev1.

This repository is divided in two subrepositories:

- FNN: enables the reproduction of the FNN results(on MNIST, CIFAR-10, SVHN)reported in the article

- RNN: enables the reproduction of the RNN results(on War and Peace, Linux Kernel) reported in the article

Requirements
This software is implemented on top of the implementation of [BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect) and has all the same requirements. 


Example training command on *War and Peace* dataset:
- training LAB
```sh
python warpeace.py --method="LAB" --lr_start=0.002 --w="w" --len=100
```
- training LAB2
```sh
python warpeace.py --method="LAB" --lr_start=0.002 --w="wa" --len=100
```

If you find loss-aware weight quantization useful in your research, please consider citing the the paper
```sh

@InProceedings{hou2017loss,
	title={Loss-aware Binarization of Deep Networks},
	author={Hou, Lu and Yao, Quanming and Kwok, James T.},
	booktitle={International Conference on Learning Representations},
	year={2017}
}

@InProceedings{hou2018loss,
	title={Loss-aware Weight Quantization of Deep Networks},
	author={Hou, Lu and Kwok, James T.},
	booktitle={International Conference on Learning Representations},
	year={2018}
}

@InProceedings{hou2019analysis,
	title={Analysis of Quantized Models},
	author={Hou, Lu and Zhang, Ruiliang and Kwok, James T.},
	booktitle={International Conference on Learning Representations},
	year={2019}
}
```
