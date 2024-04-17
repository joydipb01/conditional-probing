# Downloads example corpora.
# Includes conllx files, raw text files, and ELMo contextual word representations

# By default, downloads a (very) small subset of the EN-EWT
# universal dependencies corpus. 

wget https://nlp.stanford.edu/~johnhew/public/en_ewt-ud-sample.tgz
tar xzvf en_ewt-ud-sample.tgz
mkdir -p example/data
mv en_ewt-ud-sample example/data
rm en_ewt-ud-sample.tgz
