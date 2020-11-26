# LPR-GAN
This is used for image binarization for mainly for Vehicle Number Plate. Beside this the same architecture can be used for document binarization.

# Sample Output
Sample outcome of the experiments have been placed in the Sample_Output folder.
1. ALPR (Binarization)
   - ANPR: https://github.com/openalpr/benchmarks
   - UFPR: https://web.inf.ufpr.br/vri/databases/ufpralpr/
   - Media Lab: http://www.medialab.ntua.gr/research/LPRdatabase.html
2. Document Binarization
   - dibco17: https://vc.ee.duth.gr/dibco2017/
   - dibco18: https://vc.ee.duth.gr/h-dibco2018/
   - Palm Leaf: http://amadi.univ-lr.fr/ICFHR2016_Contest/index.php/download-123
   
# Code Description
## Files will be updated soon
1. DualDisBCD.py is the main GAN network file. 
2. img_utils.py is the post processing file which is used over the model output.
3. BCDUnetMultiScalePatchGAN.ipynb is the sample notebook file to train and test the network.

# Sample Output
The network produces very good outcome :+1:

![Alt text](Sample_Output/sampleout.PNG?raw=true "LP Binarization over various lighting condition")


# External Link
Will be updated soon

# Citation
Will be updated soon