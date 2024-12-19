from abstract2gene.dataset import bioc2dataset

# Note, all files from a single bioc archive file (i.e. all end with 0)
files = list(range(10000, 100000, 10))
dataset = bioc2dataset(files, min_occurrences=50)
dataset = bioc2dataset([0], min_occurrences=50, embed_bs=100)
dataset.save("bioc")
