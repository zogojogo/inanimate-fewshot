import torch
from torchvision import transforms, models
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

class Img2VecResnet50():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 2048
        self.modelName = "resnet-50"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getFeatureLayer(self):
        cnnModel = models.resnet50(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 2048
        return cnnModel, layer

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]

    def getEmbeddings(self, img_dir):
        Vectors = {}
        for image in tqdm(os.listdir(img_dir)):
            img = Image.open(os.path.join(img_dir, image))
            img = img.resize((224, 224))
            Vectors[image] = self.getVec(img)
            img.close()
        embeddings = np.array(list(Vectors.values())).mean(axis=0)
        return embeddings

if __name__ == "__main__":
    animate_dir = "/home/zogojogo/Nodeflux/inanimate-fewshot/animate"
    inanimate_dir = "/home/zogojogo/Nodeflux/inanimate-fewshot/inanimate"
    img2vec = Img2VecResnet50()
    anim_embeddings = img2vec.getEmbeddings(animate_dir)
    inanim_embeddings = img2vec.getEmbeddings(inanimate_dir)
    
    with open("/home/zogojogo/Nodeflux/inanimate-fewshot/embeddings.json", "w") as f:
        json.dump({"anim": anim_embeddings.tolist(), "inanim": inanim_embeddings.tolist()}, f)