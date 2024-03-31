from PIL import Image
from sklearn.datasets import load_sample_images

china = load_sample_images().images[0]
flower = load_sample_images().images[1]

imChina = Image.fromarray(china)
imFlower = Image.fromarray(flower)

imChina.show()
imFlower.show()

imChina.save('china.png')
imFlower.save('flower.png')

# im = Image.open('china.png')
# im.show()