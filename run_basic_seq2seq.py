from basic_seq2seq import preprocessing
from basic_seq2seq import training
from basic_seq2seq import masked_loss
import tensorflow as tf
import argparse

embedding_dim = 256
units = 1024

parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', 
    '--test', 
    action = "store_true", 
    help = "Keyword to make the script test on a smaller dataset",
    dest = "test_boolean"
    )
args = parser.parse_args()

print(f"Starting run_basic_seq2seq.py script!")
my_preprocessor = preprocessing.Preprocessor()


my_preprocessor.load_and_preprocess_data('spider/train_spider.json')
my_preprocessor.create_text_processors()
my_preprocessor.create_encoder_decoder()

if args.test_boolean:
    print("Testing training!")

    my_preprocessor.create_sample_dataset()
    sample_dataset = my_preprocessor.sample_dataset
    # print(f"input: {my_preprocessor.input}")
    # print(f"target: {my_preprocessor.target}")
    # print(f"sample_dataset: {sample_dataset}")
    # print(f"sample_dataset.take(1): {sample_dataset.take(1)}")
    # print(f"list(sample_dataset.take(1).as_numpy_iterator()): {list(sample_dataset.take(1).as_numpy_iterator())}")
    sample_batches = list(sample_dataset.take(1).as_numpy_iterator())
    example_input_batch, example_target_batch = sample_batches[0][0], sample_batches[0][1]

    translator = training.TrainTranslator(
        embedding_dim, units,
        input_text_processor=my_preprocessor.input_text_processor,
        output_text_processor=my_preprocessor.output_text_processor,
        use_tf_function=False)

    # Configure the loss and optimizer
    translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=masked_loss.MaskedLoss(),
    )

    for n in range(10):
        print(translator.train_step([example_input_batch, example_target_batch]))
        print()

