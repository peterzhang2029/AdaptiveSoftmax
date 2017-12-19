# Adaptive Softmax with Pytorch

This is an implementation of **Adaptive Softmax** with pytorch, the paper is "Efficient softmax approximation for GPUs"(http://arxiv.org/abs/1609.04309).

## Requirement:

python2.7 or 3 (other versions may also support)

pytorch0.2.0 (other versions may also support)

tqdm

## Train:

1. Download and prepare the data (Text8) :

    ```
    cd data
    sh makedata-text8.sh
    ```

2. Train the language model with adaptive softmax:

    ```
    python main.py
    ```

3. Train the language model with regular softmax:

    ```
    python main.py --with_adaptive 0
    ```

## Current results:

### **Test perplexity**:

| Pass |  adaptive softmax | regular softmax |
| ----- | ---------------  | --------------- |
| 1 | 227.934 | 267.424 |
| 2 | 195.780 | 226.902 |
| 3 | 182.551 | 211.505 |
| 4 | 173.464 | 199.371 |
| 5 | 168.012 | 193.128 |



### **Time cost of one sample**:

**adaptive softmax** : 16.46it/s

**regular softmax** : 4.13it/s
