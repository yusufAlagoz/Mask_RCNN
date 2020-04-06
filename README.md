
# MASK RCNN Kurulumu detaylı anlatımı aşağıda yer almaktadır.

### Mask R-CNN Nedir ?
Mask R-CNN detaylı bilgisine makalenden ulaşılabilir.https://arxiv.org/pdf/1703.06870.pdf

Video : https://www.youtube.com/watch?v=g7z4mkfRjI4

### Kurulum Aşamaları
Altyapı olarak Anaconda kullanılır. Anaconda sistemimizde kurulu değil ise detaylı olarak anlatımı aşağıdaki videodan erişilebilir.
https://www.youtube.com/watch?v=T8wK5loXkXg

### Anaconda kurulduktan sonra MaskRCNN ortamı(environment) kurulur.

conda create -n MaskRCNN python=3.6 pip

### Ortam kurulumu ile ilgili detaylı bilgi için
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### Ortam kurulduktan sonra aktifleştirilir.
source activate mrcnn

### Mask R-CNN deposu bilgisayarımıza klonlanır.
git clone https://github.com/yusufAlagoz/Mask_RCNN

Mask R-CNN için gerekli kütüphaneler depodan indiriken requirements dosyasında yer alır. Bu kütüphaneler aşağıdaki komut ile kurulur.

numpy, scipy, cython, h5py, Pillow, scikit-image, tensorflow-gpu==1.12, keras==2.1.4, jupyter

pip install -r requirements.txt

# Mask R-CNN için gerekli bir diğer kütüphane pycocotools aşağıdaki adımlar takip edilerek kurulur. 
git clone https://github.com/philferriere/cocoapi.git

use pip to install pycocotools

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

### Model eğitimin sırasında öğrenme aktarımı için kullanılcak mask_rcnn_coco.h5 dosyası indirierek depo klasörünün içerisine kopyalanır.
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

### Eğitim yapılmadan model ile ilgili demo dosyası incelenmek istenir ise ;

Bir jupyter notebook sayfası açılırak aşağıdaki komut yazılır;

Öncelikle bir terminal ekranı açılır.

jupyter notebook komutu yazılır.

Açılan sayfadan depo dosyası içerisindeki samples klasörü içerisindeki demo.ipynb dosyası açılır.


### jupyter kullanımı için detaylı anlatıma https://www.codecademy.com/articles/how-to-use-jupyter-notebooks linkinden ulaşılabilir.
Ayrıca Google 'ın  ücretsiz olarak sunduğu https://colab.research.google.com/ kullanılabilir.


### Ayrıca Mask R-CNN detaylı adım adım bilgilendirme için aşağıdaki siteye bakılabilir.
https://github.com/matterport/Mask_RCNN

