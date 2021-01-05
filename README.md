# LPR-GAN
This is used for image binarization for mainly for Vehicle Number Plate. Beside this the same architecture can be used for document binarization.

# Sample Output
Sample outcome of the experiments have been placed in the Sample_Output folder.
1. ALPR (Binarization)
   - ANPR: https://github.com/openalpr/benchmarks
   - UFPR: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
   - Media Lab: http://www.medialab.ntua.gr/research/LPRdatabase.html
2. Document Binarization
   - dibco17: https://vc.ee.duth.gr/dibco2017/
   - dibco18: https://vc.ee.duth.gr/h-dibco2018/
   - Palm Leaf: http://amadi.univ-lr.fr/ICFHR2016_Contest/index.php/download-123
   
# Code Description
1. BMDDNet.py is the main GAN network file. 

# Sample Output
The network produces very good outcome :+1:

License Plate Segmentation from road side vehicle.
![Alt text](Sample_Output/sampleout.PNG?raw=true "LP Binarization over various lighting condition")

DIBCO 17 document binarization sample.
![Alt text](Sample_Output/dibco17.PNG?raw=true "Document Binarization")

# How to run
Use the BMDDNet class to train the generative network for producing the binarized License plate.

# Citation
Will be updated soon