# FSFusion

This is official Pytorch implementation of "[A Full-Scale Hierarchical Encoder-Decoder Network with Cascading Edge-prior for Infrared and Visible Image Fusion](https://www.sciencedirect.com/science/article/pii/S0031320323008890)"

```
@article{luo2023full,
  title={A Full-Scale Hierarchical Encoder-Decoder Network with Cascading Edge-prior for Infrared and Visible Image Fusion},
  author={Luo, Xiaoqing and Wang, Juan and Zhang, Zhancheng and Wu, Xiao-jun},
  journal={Pattern Recognition},
  pages={110192},
  year={2023},
  publisher={Elsevier}
}
```

## Framework

![framework](./images/overall_end-1.png)
<p align="center">
    <em>The overall framework of the proposed FSFusion.</em>
</p>

## To test

1. Downloading the pre-trained checkpoint from [hed_pretrained_bsds.caffemodel](https://pan.baidu.com/s/18CvOFoG3qLJtnzn33oZtng?pwd=a9hu) and putting it in `./HED`.
1. The pre-trained checkpoint is put in `./models/Final.model`.
1. The test datasets are put in `./images/`.
1. The result data_root are put in `./outputs/dataset_name`.

Then running `test.py`
