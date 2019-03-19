from PIL import Image

QUAL = 32

def compress(input, output):
    im = Image.open(input)
    rgb_im = im.convert('RGB')
    rgb_im.save(output, format='JPEG', subsampling=0, quality=QUAL)

def decompress(input, output):
    im2 = Image.open(input)
    im2.save(output)