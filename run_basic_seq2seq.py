from basic_seq2seq import preprocessing
from basic_seq2seq import training
from basic_seq2seq import masked_loss
from basic_seq2seq import translating
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

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
parser.add_argument(
    '-b', 
    '--batch_size', 
    action = "store", 
    help = "Specify batch size for training model with TensorFlow. Use a small batch size (e.g. 16) if you don't have a separate GPU.",
    dest = "arg_batch_size"
    )
args = parser.parse_args()

print(f"Starting run_basic_seq2seq.py script!")
my_preprocessor = preprocessing.Preprocessor()


my_preprocessor.load_and_preprocess_data('spider/train_spider.json')
my_preprocessor.create_text_processors()
my_preprocessor.create_encoder_decoder()

if args.test_boolean:
    print("Testing training!")

    my_preprocessor.create_dataset()
    sample_dataset = my_preprocessor.dataset
    # print(f"input: {my_preprocessor.input}")
    # print(f"target: {my_preprocessor.target}")
    # print(f"sample_dataset: {sample_dataset}")
    # print(f"sample_dataset.take(1): {sample_dataset.take(1)}")
    # print(f"list(sample_dataset.take(1).as_numpy_iterator()): {list(sample_dataset.take(1).as_numpy_iterator())}")
    
    sample_batches = list(sample_dataset.take(1).as_numpy_iterator())
    example_input_batch, example_target_batch = sample_batches[0][0], sample_batches[0][1]

    print("Initializing translator!")
    translator = training.TrainTranslator(
        embedding_dim, units,
        input_text_processor=my_preprocessor.input_text_processor,
        output_text_processor=my_preprocessor.output_text_processor,
        use_tf_function=False)

    print("Compiling translator!")
    # Configure the loss and optimizer
    translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=masked_loss.MaskedLoss(),
    )

    print("Training translator!")
    for n in range(10):
        print(translator.train_step([example_input_batch, example_target_batch]))

else:
    my_preprocessor.create_dataset(int(args.arg_batch_size))
    train_translator = training.TrainTranslator(
        embedding_dim, units,
        input_text_processor=my_preprocessor.input_text_processor,
        output_text_processor=my_preprocessor.output_text_processor)

    # Configure the loss and optimizer
    train_translator.compile(
        optimizer=tf.optimizers.Adam(),
        loss=masked_loss.MaskedLoss(),
    )

    class BatchLogs(tf.keras.callbacks.Callback):
        def __init__(self, key):
            self.key = key
            self.logs = []

        def on_train_batch_end(self, n, logs):
            self.logs.append(logs[self.key])

    batch_loss = BatchLogs('batch_loss')

    train_translator.fit(my_preprocessor.dataset, epochs=3,
                     callbacks=[batch_loss])

    print("Training ended!")

    translator = translating.Translator(
        encoder=train_translator.encoder,
        decoder=train_translator.decoder,
        input_text_processor=my_preprocessor.input_text_processor,
        output_text_processor=my_preprocessor.output_text_processor,
    )

    model_filepath = 'models/basic_seq2seq'
    print(f"Saving model to {model_filepath}!")
    tf.saved_model.save(translator, model_filepath,
                    signatures={'serving_default': translator.tf_translate})