
### MASK RCNN Kurulum ve Detaylı Anlatımı
## Mask R-CNN Nedir 
Detaylı bilgi için Makale https://arxiv.org/pdf/1703.06870.pdf

Vided : https://www.youtube.com/watch?v=g7z4mkfRjI4
Overview on how to install
Step 1: create a conda virtual environment with python 3.6
Step 2: install the dependencies
Step 3: Clone the Mask_RCNN repo
Step 4: install pycocotools
Step 5: download the pre-trained weights
Step 6: Test it

## Kurulum Aşamaları
Altyapı olarak Anaconda kullanılır.

# Anaconda kurulumu ile ilgili olarak detaylı anlatım.
https://www.youtube.com/watch?v=T8wK5loXkXg

# Anaconda kurulduktan sonra MaskRCNN ortamı kurulur.
conda create -n MaskRCNN python=3.6 pip

# Ortam kurulumu ile ilgili detaylı bilgi için
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

# Ortam kurulduktan sonra aktifleştirilir.
source activate mrcnn

# Mask R-CNN deposu bilgisayarımıza klonlanır.
git clone https://github.com/yusufAlagoz/Mask_RCNN

# Mask R-CNN için gerekli kütüphaneler depodan indiriken requirements dosyasında yer alır. Bu kütüphaneler aşağıdaki komut ile kurulur.
# numpy, scipy, cython, h5py, Pillow, scikit-image, 
#tensorflow-gpu==1.12, keras==2.1.4, jupyter
pip install -r requirements.txt

#Mask R-CNN için gerekli bir diğer kütüphane pycocotools aşağıdaki adımlar takip edilerek kurulur. 
git clone https://github.com/philferriere/cocoapi.git
use pip to install pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

#Model eğitimin sırasında öğrenme aktarımı için kullanılcak mask_rcnn_coco.h5 dosyası indirierek depo klasörünün içerisine kopyalanır.
https://github.com/matterport/Mask_RCNN/releases sayfasına girilerek

mask_rcnn_coco.h5 dosyası indirilir.

# Eğitim yapılmadan model ile ilgili demo dosyası incelenmek istenir ise ;
# Terminal ekranı açılarak bir jupyter notebook sayfası açılır.;
# jupyter kullanımı için detaylı anlatıma https://www.codecademy.com/articles/how-to-use-jupyter-notebooks linkinden ulaşılabilir.
# https://hub.mybinder.turing.ac.uk/user/jupyterlab-jupyterlab-demo-e5xktmcf/lab detaylı usermanual
# Ayrıca Google 'ın  ücretsiz olarak sunduğu https://colab.research.google.com/ kullanılabilir.
jupyter notebook 

Açılan sayfadan depo dosyası içerisindeki samples klasörü içerisindeki demo.ipynb dosyası açılır.

## Ayrıca Mask R-CNN detaylı adım adım bilgilendirme için aşağıdaki siteye bakılabilir.
https://github.com/matterport/Mask_RCNN

