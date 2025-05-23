[IPMI 2025] GeoT: Geometry-guided Instance-dependent Transition Matrix for Semi-supervised Tooth Point Cloud Segmentation


### Environment
```bash
git clone https://github.com/CUHK-AIM-Group/GeoT.git
cd GeoT
conda create -n GeoT python=3.7 numpy=1.20 numba
conda activate GeoT
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt
pip install open3d einops opencv-python
pip install timm==0.5.4

cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../pointops
python setup.py install
cd ../subsampling
python setup.py install
cd ../../..
cd pointops/
python setup.py install
cd ..
cd pointnet2/
python setup.py install
cd ..
```

### Train

```
python examples/segmentation/train.py --cfg cfgs/tooth_semi/transformer_finetune_fixmatch_ntm.yaml
```