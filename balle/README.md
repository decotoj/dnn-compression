Adapted from image compression algorithm by Balle et al as implemented
by jonycgn (see: https://github.com/tensorflow/compression)

Use:

Example: Compress an Image 

'python bls2017.py compress example.png compressed.bin'



Example: Decompress an Image 

'python bls2017.py decompress compressed.bin reconstructed.png'



Example: Train a model

'python bls2017.py -v --train_glob="images/*.png" train'
