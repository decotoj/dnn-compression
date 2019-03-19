Much of this code originally adapted from image compression algorithm by Balle et al as implemented
by jonycgn (see: https://github.com/tensorflow/compression).  

Use:

Example: Compress an Image 

'python bls2017.py compress example.png compressed.bin'

Example: Compress and Image for Li w/ Optional Extra Argument to Denote Image Map Instead of Image

'python li_001.py compress example.png compressed_map.bin map'


Example: Decompress an Image 

'python bls2017.py decompress compressed.bin reconstructed.png'



Example: Train a model for baseline balle algorithm

'python bls2017.py train'


Example: Train a model for li implementation 001

'python li_001.py train'



