# PRP
This is the implementation of our paper "Video Playback Rate Perception for Self-supervised Spatio-Temporal Representation Learning".

### Environments
* Ubuntu 16.04
* Python 3.6.1
* Pytorch 0.4.1

### Prerequisits
1. Clone the repository to your local machine.

    ```
    $ git clone https://github.com/yuanyao366/PRP.git
    ```

2. Install the python dependency packages.
    
    ```
    $ pip install -r requirements.txt
    ```

### Training and Testing
1. Run `train_predict.py` to pre-train the model on the proxy task (PRP).
2. Run `ft_classfy.py` to fine-tune the model on the target task.
3. Run `test_classify` to evaluate performance on the target task.


### Pre-trained models
Pre-trained PRP model on the split1 of UCF101: 
C3D[*(OneDrive)*](https://1drv.ms/u/s!Al-IKnCwKkpqilawzdPyCbeVVjD_?e=4OycfF); R3D[*(OneDrive)*](https://1drv.ms/u/s!Al-IKnCwKkpqilocEjNpxrLY326F?e=W9LI8y); R(2+1)D[*(OneDrive)*](https://1drv.ms/u/s!Al-IKnCwKkpqiljBkCPn0nALy1H4?e=dSdnAd)