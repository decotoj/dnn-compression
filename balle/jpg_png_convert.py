from PIL import Image

im = Image.open("/home/jake/Desktop/CS230/Project/results/2092.png")
rgb_im = im.convert('RGB')
rgb_im.save('/home/jake/Desktop/CS230/Project/results/2092_jpeg_15.jpg', format='JPEG', subsampling=0, quality=15)

im2 = Image.open("/home/jake/Desktop/CS230/Project/results/2092_jpeg_15.jpg")
im2.save('/home/jake/Desktop/CS230/Project/results/2092_jpeg_15.png')