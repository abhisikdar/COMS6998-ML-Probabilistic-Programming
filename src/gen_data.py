import os
import tqdm
import zipfile
import numpy as np
import argparse
import pickle
import json
import requests
import shutil

#data = https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/antonym-synonym-dataset/
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DATA_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/antonym-synonym-dataset/ant_syn_pairs.zip"

def fetch_embeddings_file(target_file, target_dir, url=GLOVE_URL, delete_zip=True):
    #if the dataset already exists exit
    if os.path.isfile(target_file):
        print("Embeddings already downloded")
        return

    # download (large) zip file
    # for large https request on stream mode to avoid out of memory issues
    # see : http://masnun.com/2016/09/18/python-using-the-requests-module-to-download-large-files-efficiently.html
    print("**************************")
    print("  Downloading zip file")
    print("  >_<  Please wait >_< ")
    print("**************************")
    response = requests.get(url, stream=True)
    #read chunk by chunk
    handle = open('glove.zip', "wb")
    for chunk in tqdm.tqdm(response.iter_content(chunk_size=512)):
        if chunk:  
            handle.write(chunk)
    handle.close()  
    print("  Download completed ;) :") 
    #extract zip_file
    zf = zipfile.ZipFile('glove.zip')
    print("Extracting {} file".format('glove.zip'))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    zf.extractall(target_dir)
    if delete_zip:
        print("Deleting {} file".format("glove.zip"))
        os.remove(path="glove.zip")

def download_dataset(target_dir, url=DATA_URL, delete_zip=True):
    #if the dataset already exists exit
    if os.path.exists(target_dir):
        count = 0
        for f in os.listdir(target_dir):
            path = os.path.join(target_dir, f)
            if not os.path.isdir(path):
                count += len(open(path, "r").readlines())
        if count == 15632:
            print("datasets already downloded")
            return
        print("All files not found/Corrupted download; downloading again")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    response = requests.get(url, stream=True)
    handle = open('data.zip', "wb")
    for chunk in tqdm.tqdm(response.iter_content(chunk_size=512)):
        if chunk:  
            handle.write(chunk)
    handle.close() 
    print("  Dataset download completed ;) :")
    #extract zip_file
    
    zf = zipfile.ZipFile('data.zip')
    print("Extracting {} file".format('data.zip'))
    zf.extractall()
    source_dir = "./eacl2017"
    file_names = os.listdir(source_dir)
    
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)

    if delete_zip:
        print("Deleting {} file".format("data.zip"))
        os.remove(path="data.zip")
        os.rmdir(path=source_dir)

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
                        help='Full path to the dataset directory containing all synonym and antonym files')
    parser.add_argument('--embedding-dir', dest='embedding_dir', required=True,
                        help='Full path to the directory containing glove embeddings')
    parser.add_argument('--outdir', dest='outdir', required=True,
                        help='Full path to the output directory where plot is saved')

    args = parser.parse_args()

    glove_file = os.path.join(args.embedding_dir, "glove.6B.100d.txt")

    fetch_embeddings_file(glove_file, args.embedding_dir)
    
    download_dataset(args.data_dir)
    
    embeddings = fetch_embeddings(glove_file)

    word1_embeddings, word2_embeddings, y, word_list, embedding_matrix, word1_list, word2_list = load_syn_ant_dset(args.data_dir, embeddings, 100)

    data = json.dumps(
        {   
            "N": len(y),
            "D": len(word1_embeddings[0]),
            "v1": word1_embeddings, 
            "v2": word2_embeddings, 
            "y": y
        }
    )

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("Generating data for stan model")
    with open(os.path.join(args.outdir, "data.json"), "w") as write_handle:
        write_handle.write(data)
    print("Generated files: " + os.path.join(args.outdir, "data.json"))
        
    print("Generating embedding matrix")
    with open(os.path.join(args.outdir, "embedding_matrix.pkl"), "wb") as write_handle:
        pickle.dump(embedding_matrix, write_handle)
    print("Generated files: " + os.path.join(args.outdir, "embedding_matrix.pkl"))
    
    print("Saving the complete list of words for our models")
    with open(os.path.join(args.outdir, "word_list.pkl"), "wb") as write_handle:
        pickle.dump(word_list, write_handle)
    print("Generated files: " + os.path.join(args.outdir, "word_list.pkl"))

    with open(os.path.join(args.outdir, "pairs.pkl"), "wb") as write_handle:
        pickle.dump(np.hstack((np.array(word1_embeddings), np.array(word2_embeddings), np.array(y).reshape(len(y), 1))), write_handle)
    
    word_pairs = []
    for i in range(len(word1_list)):
            word_pairs.append((word1_list[i], word2_list[i]))
            
    with open(os.path.join(args.outdir, "word_pairs.pkl"), "wb") as write_handle:
        pickle.dump(word_pairs, write_handle)
