# AI-Image-Classifier-with-ResNet50

## Description
This project uses the ResNet50 model to extract features from images and search for similar images using FAISS (Facebook AI Similarity Search). The project supports building an image index from a folder of product images and finding similar images based on extracted features.

## Project Structure

- `image_search.py`: The `ImageSearch` class for extracting image features, building an index, adding images to the index, and searching for similar images.
- `config.py`: Configuration file for paths and parameters.
- `main.py`: Main file to run the key functionalities of the project (build the index and search for images).
- `test_image`: Folder containing test images for model evaluation.

## Installation

### Requirements
This project requires the following Python libraries:

- TensorFlow 2.x
- NumPy
- FAISS
- tqdm

You can install all dependencies from the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

### Build the Image Index
To build an index from images in the folder, you can use the following code in main.py:
```python
from image_search import ImageSearch

# Create an ImageSearch object and build the index
searcher = ImageSearch()
searcher.build_index()
```
<img src="https://github.com/HitDrama/AI-Image-Classifier-with-ResNet50/blob/main/static/test/test.png" alt="Sample Image" />


### Developer
Dang To Nhan

