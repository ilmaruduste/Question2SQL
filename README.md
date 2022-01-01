# Question2SQL
An NLP project for translating natural language questions into SQL queries using different methods. The Seq2Seq with Attention model used here is mostly based off of [Tensorflow tutorial for NMTs with attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention).

A technical report is available in the repo under `question2sql_report.pdf`.

If you want to run this Question2SQL yourself, then I suggest using Google Colab. A prefilled Google Colab notebook is available [here](https://colab.research.google.com/drive/1jnamy1Rm_HqOWXAO-wf7k8rd5WpEJK03?usp=sharing). From beginning to end, a Google Colab notebook will run about 30 minutes on the Spider dataset and 50 minutes on WikiSQL, depending on the number of epochs you want it to train for.

Before running basic_seq2seq, be sure to download the necessary reference data (e.g. Spider, WikiSQL, pre-trained GloVe embeddings) via the bash scripts in `ref_data_download_scripts`:
```
./ref_data_download_scripts/download_spider.sh
./ref_data_download_scripts/download_readable_wikisql.sh
./ref_data_download_scripts/download_pretrained_glove.sh
```
And then run the basic Seq2Seq script for trainng and saving a Seq2Seq model on the Spider dataset. 
The `-b` argument is for batch size and the `-c` argument is for whether or not table and column names should be concatenated to the input queries while training. `-e` is used for specifying the epoch number for training and use `-wiki` if you want to run it on WikiSQL and not Spider data (Spider is the default). NB! Concatenation does not work on the WikiSQL dataset!
```
python run_basic_seq2seq.py -b 8 -c
```
After training a model, you can test it via the `test_basic_seq2seq.py` script. 
The `-m` argument is the model filepath, the `-f` argument is the filepath for the results as a .csv file and the `-c` argument is (like with training) for concatenating table and column names to input queries.
```
python test_basic_seq2seq.py -m models/basic_seq2seq -f results/basic_seq2seq_32_concatenated.csv -c
```
