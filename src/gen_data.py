import os
import tqdm
import zipfile
import nltk
import numpy as np
import argparse
import pickle
import json
import requests

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

def fetch_embedding_matrix(embedding_dict, dim):
    wordlist = []
    res = np.zeros((len(embedding_dict), dim), dtype='float32')
    i = 0
    for word in sorted(embedding_dict.keys()):
        wordlist.append(word)
        res[i] = embedding_dict[word]
        i += 1

    return res, wordlist

def load_syn_ant_dset(dir, embedding_dict, dim):
    v1_embeds = []
    v2_embeds = []
    y = []
    word_list = set()
    word1_list = []
    word2_list = []

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
                    word_list.add(word1)
                    word_list.add(word2)
                    word1_list.append(word1)
                    word2_list.append(word2)
    
    embedding_matrix = np.zeros((len(word_list), dim), dtype='float32')
    for i, word in enumerate(list(word_list)):
        embedding_matrix[i] = embedding_dict[word]
    
    return v1_embeds, v2_embeds, y, list(word_list), embedding_matrix, word1_list, word2_list




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

    word1_embeddings, word2_embeddings, y, word_list, embedding_matrix, word1_list, word2_list = load_syn_ant_dset(args.data_dir, embeddings, 100)

    # embedding_matrix, word_list = fetch_embedding_matrix(embeddings, 100)

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
        
    with open(os.path.join(args.outdir, "embedding_matrix.pkl"), "wb") as write_handle:
        pickle.dump(embedding_matrix, write_handle)
    
    with open(os.path.join(args.outdir, "word_list.pkl"), "wb") as write_handle:
        pickle.dump(word_list, write_handle)
    
    with open(os.path.join(args.outdir, "word_list.pkl"), "wb") as write_handle:
        pickle.dump(word_list, write_handle)
    
    with open(os.path.join(args.outdir, "pairs.pkl"), "wb") as write_handle:
        pickle.dump(np.hstack((np.array(word1_embeddings), np.array(word2_embeddings), np.array(y).reshape(len(y), 1))), write_handle)
    
    word_pairs = []
    for i in range(len(word1_list)):
            word_pairs.append((word1_list[i], word2_list[i]))
            
    with open(os.path.join(args.outdir, "word_pairs.pkl"), "wb") as write_handle:
        pickle.dump(word_pairs, write_handle)
