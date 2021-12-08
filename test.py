import json
import argparse
from embeddings.embedding_extract import Img2VecResnet50
from numpy import dot
from numpy.linalg import norm
from PIL import Image
import matplotlib.pyplot as plt

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def get_similarity(img_path):
    img2vec = Img2VecResnet50()
    img = Image.open(img_path).convert("RGB")
    img_vec = img2vec.getVec(img)

    with open("embeddings/embeddings.json", "r") as f:
      embeddings = json.load(f)
    anim_embeddings = embeddings["anim"]
    inanim_embeddings = embeddings["inanim"]

    result = [cosine_similarity(anim_embeddings, img_vec), cosine_similarity(inanim_embeddings, img_vec)]
    return result

def get_output(img_path):
    result = get_similarity(img_path)
    if result[0] > result[1]:
        return "animate", result[0]
    else:
        return "inanimate", result[1]

def visualize_result(img_path):
    img = Image.open(img_path).convert("RGB")
    result, confidence = get_output(img_path)
    plt.imshow(img)
    plt.title("{} ({:.2f}%)".format(result, confidence * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help= "Input image path", type=str, required=True)
    args = parser.parse_args()
    visualize_result(args.img)
    plt.show()