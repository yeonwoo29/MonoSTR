

## Installation
    ```bash

    conda create -n monostr python=3.8
    conda activate monostr

    
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

    pip install -r requirements.txt

    cd lib/models/monostr/ops/
    bash make.sh
    
    cd ../../../..
    ```
 
    You can also change the data path at "dataset/root_dir" in `configs/monostr.yaml`.
    

### Train
  ```bash
  bash train.sh configs/monostr.yaml > logs/monostr.log
  ```
### Test
  ```bash
  bash test.sh configs/monostr.yaml
  ```