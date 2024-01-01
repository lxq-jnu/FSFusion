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
1.The pre-trained checkpoint is put in **./models/Final.model**.

2.The test datasets are put in **./images/**.

3.The result data_root are put in **./outputs/dataset_name**.

Then running test.py
