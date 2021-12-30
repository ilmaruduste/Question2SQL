from basic_seq2seq import preprocessing
from basic_seq2seq import translating
import tensorflow as tf
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', 
    '--resultsfilepath', 
    action = "store", 
    help = "Filepath to save ",
    dest = "results_filepath"
    )
parser.add_argument(
    '-m', 
    '--model', 
    action = "store", 
    help = "Specify the filepath of the model to test.",
    dest = "model"
    )
parser.add_argument(
    '-c', 
    '--concatenate', 
    action = "store_true", 
    help = "Concatenate input together with table name and table columns for a potentially better input.",
    dest = "concatenate"
    )
parser.add_argument(
    '-wiki', 
    '--wikisql', 
    action = "store_true", 
    help = "If true, process and test WikiSQL dataset.",
    dest = "wiki",
    default = False
    )

args = parser.parse_args()

print(f"Starting test_basic_seq2seq.py script!")
my_preprocessor = preprocessing.Preprocessor()

if args.wiki:
    my_preprocessor.load_and_preprocess_data('wikisql/test.csv', args.concatenate, wikisql = True)
else:
    my_preprocessor.load_and_preprocess_data('spider/dev.json', args.concatenate)

translator = tf.saved_model.load(args.model)
# print(translator.summary())

# without_preprocessing_result = translator.translate(my_preprocessor)
with_preprocessing_result = translator.tf_translate(my_preprocessor.input)
model_output_list = [query.numpy().decode() for query in list(with_preprocessing_result['text'])]
# print(model_output_list)

results_dict = dict({'original_input': my_preprocessor.input, 
    'model_output': model_output_list, 
    'original_output': my_preprocessor.target})

for i in range(len(my_preprocessor.input)):
    print(f"original_input: {results_dict['original_input'][i]}")
    print(f"model_output: {model_output_list[i]}")
    print(f"original_output: {results_dict['original_output'][i]}")
    print()

if args.results_filepath:
    print("Creating results dataframe!")
    results = pd.DataFrame(results_dict)
    print(f"Saving results dataframe to {args.results_filepath}!")
    results.to_csv(args.results_filepath, sep = ';')