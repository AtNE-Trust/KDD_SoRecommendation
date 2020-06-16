
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 500;

namespace {

bool ScanWord(StringPiece* input, string* word) {
  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;
  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace

class SkipgramSideOp : public OpKernel {
 public:
  explicit SkipgramSideOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

    mutex_lock l(mu_);
    label_pos_ = kSentenceSize;
    example_pos_ = kSentenceSize;
    sentence_pos_ = 2 * corpus_size_;
    for (int i = 0; i < kPrecalc; ++i) {
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label,
                  &precalc_examples_[i].pos, &precalc_examples_[i].neg);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    Tensor num_pos(DT_INT32, TensorShape({batch_size_}));
    auto Tnum_pos = num_pos.flat<int32>();
    Tensor num_neg(DT_INT32, TensorShape({batch_size_}));
    auto Tnum_neg = num_neg.flat<int32>();
    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        Tnum_pos(i) = precalc_examples_[precalc_index_].pos;
        Tnum_neg(i) = precalc_examples_[precalc_index_].neg;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input,
                        &precalc_examples_[j].label,
                        &precalc_examples_[j].pos,
                        &precalc_examples_[j].neg);
          }
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }
    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
    ctx->set_output(7, num_pos);
    ctx->set_output(8, num_neg);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
    int32 pos;
    int32 neg;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor freq_;
  int64 corpus_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<Example> precalc_examples_;
  std::vector<int32> sentence_;
  int precalc_index_ = 0;

  mutex mu_;
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int32 sentence_pos_ GUARDED_BY(mu_);
  int32 example_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  int32 num_pos_ GUARDED_BY(mu_) = 0;
  int32 num_neg_ GUARDED_BY(mu_) = 0;

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ wraps around at the end of corpus_.
  void NextExample(int32* example, int32* label, int32* pos, int32* neg) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (label_pos_ >= kSentenceSize || sentence_[label_pos_ - 1] == -3
       || label_pos_ - example_pos_ > 2 * window_size_ || sentence_[example_pos_ + 1] == -3) {
        ++total_words_processed_;
        num_pos_ = 0;
        num_neg_ = 0;
        example_pos_ += 2;
        if (example_pos_ >= kSentenceSize) {
          example_pos_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++sentence_pos_) {
            if (sentence_pos_ >= 2 * corpus_size_) {
              ++current_epoch_;
              sentence_pos_ = 0;
            }
            sentence_[i] = corpus_[sentence_pos_];
          }
        }
        label_pos_ = example_pos_;
      }
      if (example_pos_ != label_pos_) {
        break;
      }
      label_pos_ += 2;
    }
    *example = sentence_[example_pos_];
    *label = sentence_[label_pos_];

    if (sentence_[label_pos_ - 1] == -1) {
      ++num_pos_;
    } else {
      ++num_neg_;
    }

    *pos = num_pos_;
    *neg = num_neg_;
    label_pos_ += 2;
  }

  Status Init(Env* env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    StringPiece input = data;
    string w;
    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      if (w.compare("&") != 0 && w.compare("+") != 0 && w.compare("-") != 0) {
        ++(word_freq[w]);
        ++corpus_size_;
      }
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      ordered.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
              << " bytes, " << corpus_size_ << " words, " << ordered.size() << " unique words.";
    word_freq.clear();
    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(ordered.size());
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    std::unordered_map<string, int32> word_id;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      word.flat<string>()(i) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(i) = word_count;
      word_id[w] = i;
    }
    word_id["+"] = -1;
    word_id["-"] = -2;
    word_id["&"] = -3;
    word_ = word;
    freq_ = freq;
    corpus_.reserve(2 * corpus_size_);
    input = data;
    while (ScanWord(&input, &w)) {
      corpus_.push_back(gtl::FindWithDefault(word_id, w, -4));
    }
    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("SkipgramSide").Device(DEVICE_CPU), SkipgramSideOp);

class NegTrainSideOp : public OpKernel {
 public:
  explicit NegTrainSideOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));

    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainSideOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                errors::InvalidArgument("w_in.shape == w_out.shape"));
    Tensor b_in_pos = ctx->mutable_input(2, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(b_in_pos.shape()),
                errors::InvalidArgument("Must be a vector"));
    OP_REQUIRES(ctx, b_in_pos.dim_size(0) == w_in.dim_size(0),
                errors::InvalidArgument("b_in_pos.dim_size(0) == w_in.dim_size(0)"));
    Tensor b_in_neg = ctx->mutable_input(3, false);
    OP_REQUIRES(ctx, b_in_pos.shape() == b_in_neg.shape(),
                errors::InvalidArgument("b_in_pos.shape == b_in_neg.shape"));
    Tensor b_out_pos = ctx->mutable_input(4, false);
    OP_REQUIRES(ctx, b_in_pos.shape() == b_out_pos.shape(),
                errors::InvalidArgument("b_in_pos.shape == b_out_pos.shape"));
    Tensor b_out_neg = ctx->mutable_input(5, false);
    OP_REQUIRES(ctx, b_in_pos.shape() == b_out_neg.shape(),
                errors::InvalidArgument("b_in_pos.shape == b_out_neg.shape"));
    const Tensor& examples = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& labels = ctx->input(7);
    OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                errors::InvalidArgument("examples.shape == labels.shape"));
    const Tensor& learning_rate = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));
    const Tensor& multiplier = ctx->input(9);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(multiplier.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& lambda = ctx->input(10);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lambda.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Tb_in_pos = b_in_pos.flat<float>();
    auto Tb_in_neg = b_in_neg.flat<float>();
    auto Tb_out_pos = b_out_pos.flat<float>();
    auto Tb_out_neg = b_out_neg.flat<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto Tmul = multiplier.flat<float>();
    auto lr = learning_rate.scalar<float>()();
    auto lam = lambda.scalar<float>()();
    const int64 vocab_size = w_in.dim_size(0);
    const int64 dims = w_in.dim_size(1);
    const int64 batch_size = examples.dim_size(0);
    OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                        " vs. ", sampler_->num()));

    // Gradient accumulator for v_in.
    Tensor buf(DT_FLOAT, TensorShape({dims}));
    auto Tbuf = buf.flat<float>();

    // Scalar buffer to hold sigmoid(+/- dot).
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    // The following loop needs 2 random 32-bit values per negative
    // sample.  We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    for (int64 i = 0; i < batch_size; ++i) {
      const int32 example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example << Texamples << Tlabels;
      const int32 label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label << Texamples << Tlabels;
      const float mul = Tmul(i);
      auto v_in = Tw_in.chip<0>(example);

      if (mul > 0) {
        // Positively linked nodes: both positive and negative update
        auto b_in = Tb_in_pos(example);
        auto b_out = Tb_out_pos(label);

        // Positive: example predicts label.
        //   forward: x = v_in' * v_out + b_in + b_out
        //            l = - mul * log(sigmoid(x)) + lam / 2 * b^2
        //   backward: dl/dx = - mul * g = - mul * sigmoid(-x)
        //             dl/d(v_in) = - mul * g * v_out'
        //             dl/d(v_out) = - mul * v_in' * g
        //             dl/d(b_in) = - mul * g + lam * b_in
        //             dl/d(b_out) = - mul * g + lam * b_out
        {
          auto v_out = Tw_out.chip<0>(label);
          auto dot = (v_in * v_out).sum() + b_in + b_out;
          g = (dot.exp() + 1.f).inverse() * lr * mul;
          Tbuf = v_out * g();
          v_out += v_in * g();
          Tb_in_pos(example) *= (1 - lam * lr);
          Tb_in_pos(example) += g();
          Tb_out_pos(label) *= (1 - lam * lr);
          Tb_out_pos(label) += g();
        }
      } else {
        // Negatively linked nodes: only positive update
        auto b_in = Tb_in_neg(example);
        auto b_out = Tb_out_neg(label);

        // Positive: example predicts label.
        //   forward: x = - v_in' * v_out + b_in + b_out
        //            l = mul * log(sigmoid(x)) + lam / 2 * b^2
        //   backward: dl/dx = mul * g = mul * sigmoid(-x)
        //             dl/d(v_in) = - mul * g * v_out'
        //             dl/d(v_out) = - mul * v_in' * g
        //             dl/d(b_in) = mul * g + lam * b_in
        //             dl/d(b_out) = mul * g + lam * b_out
        {
          auto v_out = Tw_out.chip<0>(label);
          auto dot = (v_in * v_out).sum() - b_in - b_out;
          g = ((-dot).exp() + 1.f).inverse() * lr * mul;
          Tbuf = v_out * g();
          v_out += v_in * g();
          Tb_in_neg(example) *= (1 - lam * lr);
          Tb_in_neg(example) += -g();
          Tb_out_neg(label) *= (1 - lam * lr);
          Tb_out_neg(label) += -g();
        }
      }

      auto b_in_pos = Tb_in_pos(example);
      auto b_in_neg = Tb_in_neg(example);
      // Negative samples:
      //   forward: x = - v_in' * v_sample
      //            l = - log(sigmoid(x))
      //   backward: dl/dx = -g = -sigmoid(-x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      for (int j = 0; j < num_samples_; ++j) {
        const int sample = sampler_->Sample(&srnd);
        if (sample == label) continue;  // Skip.
        auto v_sample = Tw_out.chip<0>(sample);
        auto dot = (v_in * v_sample).sum();
        g = -((-dot).exp() + 1.f).inverse() * lr;
        Tbuf += v_sample * g();
        v_sample += v_in * g();
      }

      // Applies the gradient on v_in.
      v_in += Tbuf;
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainSide").Device(DEVICE_CPU), NegTrainSideOp);

}  // end namespace tensorflow
