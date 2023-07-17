# Fine-grained weed recognition
## Fine-grained weed recognition using Swin Transformer and two-stage transfer learning
__A fine-grained recognition method based on Swin Transformer with a contrastive loss was proposed for weed recognition.__

## Pretrained Models
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

## Training and testing dataset
root
    ---train      # MWFI, orginal training data
    ---test       # MWFI, testing data

## Requirements
pip install -r requirements.txt

## Run
python train_c.py
    
## Citation
If you find this resource helpful, please cite.

```
@article{,
  title={Fine-grained weed recognition using Swin Transformer and two-stage transfer learning},
  author={Yecheng Wang, Shuangqing Zhang, Baisheng Dai*, Sensen Yang , Haochen Song},
  journal={Frontiers in Plant Science},
  volume={14},
  pages={1134932},
  year={2023},
}
```
