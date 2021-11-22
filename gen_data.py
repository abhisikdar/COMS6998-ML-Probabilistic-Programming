from nltk.corpus import wordnet

import os
import tqdm
import requests
import zipfile
import nltk
import numpy as np
import argparse
import pickle
import json

#data = https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/antonym-synonym-dataset/
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
URL = "http://nlp.stanford.edu/data/glove.6B.zip"

def fetch_data(url=URL, target_file='glove.6B.zip', delete_zip=False):
    #if the dataset already exists exit
    if os.path.isfile(target_file):
        print("datasets already downloded")
        return

    #download (large) zip file
    #for large https request on stream mode to avoid out of memory issues
    #see : http://masnun.com/2016/09/18/python-using-the-requests-module-to-download-large-files-efficiently.html
    print("**************************")
    print("  Downloading zip file")
    print("  >_<  Please wait >_< ")
    print("**************************")
    response = requests.get(url, stream=True)
    #read chunk by chunk
    handle = open(target_file, "wb")
    for chunk in tqdm.tqdm(response.iter_content(chunk_size=512)):
        if chunk:  
            handle.write(chunk)
    handle.close()  
    print("  Download completed ;) :") 
    #extract zip_file
    zf = zipfile.ZipFile(target_file)
    print("1. Extracting {} file".format(target_file))
    zf.extractall()
    if delete_zip:
        print("2. Deleting {} file".format(dataset_name+".zip"))
        os.remove(path=zip_file)

EMBEDDING_VECTOR_LENGTH = 50 # <=200

def fetch_embeddings(glove_file):
    embedding_dict = {}
    with open(glove_file,'r') as f:
        for line in f:
            values=line.split()
            # get the word
            word=values[0]
            vector = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vector

    return embedding_dict


def load_syn_ant_dset(dir, embedding_dict):
    v1_embeds = []
    v2_embeds = []
    y = []

    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        with open(filepath, 'r') as f:
            pairs = f.readlines()
            for pair in tqdm.tqdm(pairs):
                word1, word2, label = pair.split('\t')
                if word1 in embedding_dict and word2 in embedding_dict:
                    v1_embeds.append(embedding_dict[word1].tolist())
                    v2_embeds.append(embedding_dict[word2].tolist())
                    y.append(int(label.strip('\n')))
    
    return v1_embeds, v2_embeds, y




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', required=True,
                        help='Full path to the dataset directory containing all synonym and antonym files and glove embeddings')
    parser.add_argument('--outdir', dest='outdir', required=True,
                        help='Full path to the output directory where plot is saved')

    args = parser.parse_args()

    # fetch_data()

    glove_file = "glove.6B.100d.txt"

    embeddings = fetch_embeddings(glove_file)

    word1_embeddings, word2_embeddings, y = load_syn_ant_dset(args.data_dir, embeddings)
    data = json.dumps(
        {   
            "N": len(y),
            "D": len(word1_embeddings[0]),
            "v1": word1_embeddings, 
            "v2": word2_embeddings, 
            "y": y
        }
    )
    
    with open(os.path.join(args.outdir, "data.json"), "w") as write_handle:
         write_handle.write(data)
