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
args = parser.parse_args()

print(f"Starting test_basic_seq2seq.py script!")
my_preprocessor = preprocessing.Preprocessor()

my_preprocessor.load_and_preprocess_data('spider/dev.json')

translator = tf.saved_model.load(args.model)
# print(translator.summary())

# without_preprocessing_result = translator.translate(my_preprocessor)
with_preprocessing_result = translator.tf_translate(my_preprocessor.input)
print(with_preprocessing_result)

results_dict = dict({'original_input': my_preprocessor.input, 
    'model_output': list(with_preprocessing_result['text']), 
    'original_output': my_preprocessor.target})

for i in range(len(my_preprocessor.input)):
    print(f"original_input: {results_dict['original_input'][i]}")
    print(f"model_output: {results_dict['model_output'][i]}")
    print(f"original_output: {results_dict['original_output'][i]}")
    print()

if args.results_filepath:
    print("Creating results dataframe!")
    results = pd.DataFrame(results_dict)
    print(f"Saving results dataframe to {args.results_filepath}!")
    results.to_csv(args.results_filepath, sep = ';')