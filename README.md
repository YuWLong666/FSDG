# Fine-Grained Domain Generalization with Feature Structuralization

In this repository, we provide the implementation of the following IEEE TMM 2025 paper: "Fine-Grained Domain Generalization with Feature Structuralization". 
The PDF of the paper is available at https://arxiv.org/pdf/2406.09166.



## Prerequisites:

* Python3
* PyTorch == 1.13.1 (with suitable CUDA and CuDNN version)
* torchvision >= 0.14.1
``` linux
pip install -r requirements.txt
```

## Download Datasets:

The CUB-Paintings dataset contains two sub-datasets, CUB-200-2011 and CUB-200-Paintings. You need to download the data for both domains.
Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
Reference https://github.com/thuml/PAN for CUB-200-Paintings.

The CompCars dataset includes two sub-datasets, Web and Surveillance, from two different domains.

The Birds-31 dataset has three domains: CUB-200-2011, NABirds, and iNaturalist2017.

## Prepare Datasets:


Firstly, link the dataset directory to the ./data directory. e.g. 

``` linux
ln -s /path/to/dataset/ ./data/
```

Secondly, update the image paths in each .txt file within the ./dataset_list directory. 

``` Python
python data_list.py
```


Then, you will see two files in the folder of ./data_list.

Note that you may need to modify the data_list.py file to match the directory structure of your dataset. 
Feel free to modify it as needed, expecially line 373 in the data_list.py file.
For now, the code supports the CUB-Paintings datasets.
The code of modifying the other two datasets are masked in data_list.py. 
You can use those codes as reference.


## Training on one dataset:

You can use the following commands to execute the training:

``` Python
python train.py --b_bkb_c false --dataset cp2 -b_po -c_po 1 -b_ssdgc -c_ssdgc 0.05 -b_dssgp -c_dssgp 1 -b_dsdgc -c_dsdgc 0.1 -ot db-training
```
where --b_bkb_c controls whether to use double backbones, 
--dataset specifies the dataset to use. 
-b_po, -c_po, -b_ssdgc, -c_ssdgc, -b_dssgp, -c_dssgp, -b_dsdgc, -c_dsdgc control the hyperparameters of alignment functions. 

Please configure the parameters based on your specific requirements.

## Citation:

If you use this code for your research, please consider citing:

```
@article{yu2024fine,
  title={Fine-Grained Domain Generalization with Feature Structuralization},
  author={Yu, Wenlong and Chen, Dongyue and Wang, Qilong and Hu, Qinghua},
  journal={arXiv preprint arXiv:2406.09166},
  year={2024}
}
```
## Contact
If you have any problem about our code, feel free to contact wlong yu@126.com.
We greatly thank the code contributors of PAN (Cited in the main paper) and appreciate any other feedbacks.