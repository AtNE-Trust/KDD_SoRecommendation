
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("SkipgramSide")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .Output("num_pos: int32")
    .Output("num_neg: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Doc(R"doc(
Parses a text file and creates a batch of examples.

vocab_word: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
words_per_epoch: The number of words in each epoch used for learning rate decaying.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
num_pos: A vector of positive links between examples and labels.
num_neg: A vector of negative links between examples and labels.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the right of the target.
)doc");

REGISTER_OP("NegTrainSide")
    .Input("w_in: Ref(float)")
    .Input("w_out: Ref(float)")
    .Input("b_in_pos: Ref(float)")
    .Input("b_in_neg: Ref(float)")
    .Input("b_out_pos: Ref(float)")
    .Input("b_out_neg: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .Input("multiplier: float")
    .Input("lambda: float")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .Doc(R"doc(
Training via negative sampling.

w_in: input word embedding.
w_out: output word embedding.
b_in_pos: positive in-degree bias.
b_in_neg: negative in-degree bias.
b_out_pos: positive out-degree bias.
b_out_neg: negative out-degree bias.
examples: A vector of word ids.
labels: A vector of word ids.
lr: Learning rate.
multiplier: Multiplier determined by distance and sign.
lambda: Regularization parameter for bias terms
vocab_count: Count of words in the vocabulary.
num_negative_samples: Number of negative samples per example.
)doc");

}  // end namespace tensorflow
