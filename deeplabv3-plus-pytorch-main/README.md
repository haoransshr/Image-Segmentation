This code is modified from [bubbliiiing's github repository](https://github.com/bubbliiiing/deeplabv3-plus-pytorch) which using the MIT open source license.

## Usage
### Preparation
Install the requirements, copy datasets into paths first.<br>
Images path: 
```
./datasets/images
```
Labels path:
```
./datasets/labels
```
Run data_split.py before training, which generates the list of training, validation and testing file lists.
```shell
# the lists will be saved in ./datasets/lists/
python data_split.py
```
### Training
Run train.py
```shell
python train.py
```
The results and pre-trained model will be saved in './logs/'
### Testing
Modify test.py to specify the pre-trained model, then run test.py.
```shell
python test.py
```
The results will be saved in './miou_out/'

