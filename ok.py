from PIL import Image
from os import listdir, path

test_directory = "dataset/testing_data/images"
train_directory = "dataset/training_data/images"

images = [
    path.join(test_directory, image)
    for image in listdir(test_directory)
]
images.extend([
    path.join(train_directory, image)
    for image in listdir(train_directory)
])
print(images)

image_sizes = set()

for image in images:
    i = Image.open(image)
    image_sizes.add(i.size)

for size in image_sizes:
    print(size)
