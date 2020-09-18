## Simple PyTorch implementation of DCGAN

### Training

Runing traning of DCGAN model on MNIST dataset

```python
python main.py --action="train" --dataset="mnist"  --lr=0.0002 --epochs=100 --batch_size=128
```

![](mnist_DCGAN.gif)





Runing traning of DCGAN model on MNIST dataset

```python
python main.py --action="train" --dataset="cifar"  --lr=0.0002 --epochs=100 --batch_size=128
```

![](cifar_DCGAN.gif)

