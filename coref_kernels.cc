#include <set>
#include <map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ExtractMentions")
.Input("mention_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("num_output_mentions: int32")
.Output("output_mention_indices: int32");

class ExtractMentionsOp : public OpKernel {
public:
  explicit ExtractMentionsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstVec mention_scores = context->input(0).vec<float>();
    TTypes<int32>::ConstVec candidate_starts = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec candidate_ends = context->input(2).vec<int32>();
    int num_output_mentions = context->input(3).scalar<int32>()(0);

    int num_input_mentions = mention_scores.dimension(0);

    Tensor* output_mention_indices_tensor = nullptr;
    TensorShape output_mention_indices_shape({num_output_mentions});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_mention_indices_shape, &output_mention_indices_tensor));
    TTypes<int32>::Vec output_mention_indices = output_mention_indices_tensor->vec<int32>();

    std::vector<int> sorted_input_mention_indices(num_input_mentions);
    std::iota(sorted_input_mention_indices.begin(), sorted_input_mention_indices.end(), 0);

    std::sort(sorted_input_mention_indices.begin(), sorted_input_mention_indices.end(),
              [&mention_scores](int i1, int i2) {
                return mention_scores(i2) < mention_scores(i1);
              });
    std::vector<int> top_mention_indices;
    int current_mention_index = 0;
    while (top_mention_indices.size() < num_output_mentions) {
      int i = sorted_input_mention_indices[current_mention_index];
      bool any_crossing = false;
      for (const int j : top_mention_indices) {
        if (is_crossing(candidate_starts, candidate_ends, i, j)) {
          any_crossing = true;
          break;
        }
      }
      if (!any_crossing) {
        top_mention_indices.push_back(i);
      }
      ++current_mention_index;
    }

    std::sort(top_mention_indices.begin(), top_mention_indices.end(),
              [&candidate_starts, &candidate_ends] (int i1, int i2) {
                if (candidate_starts(i1) < candidate_starts(i2)) {
                  return true;
                } else if (candidate_starts(i1) > candidate_starts(i2)) {
                  return false;
                } else if (candidate_ends(i1) < candidate_ends(i2)) {
                  return true;
                } else if (candidate_ends(i1) > candidate_ends(i2)) {
                  return false;
                } else {
                  return i1 < i2;
                }
              });

    for (int i = 0; i < num_output_mentions; ++i) {
      output_mention_indices(i) = top_mention_indices[i];
    }
  }
private:
  bool is_crossing(TTypes<int32>::ConstVec &candidate_starts, TTypes<int32>::ConstVec &candidate_ends, int i1, int i2) {
    int s1 = candidate_starts(i1);
    int s2 = candidate_starts(i2);
    int e1 = candidate_ends(i1);
    int e2 = candidate_ends(i2);
    return (s1 < s2 && s2 <= e1 && e1 < e2) || (s2 < s1 && s1 <= e2 && e2 < e1);
  }
};


REGISTER_KERNEL_BUILDER(Name("ExtractMentions").Device(DEVICE_CPU), ExtractMentionsOp);


REGISTER_OP("ExtractMentionsCKY")
.Attr("mentions_per_word: float")
.Input("mention_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("sentence_lengths: int32")
.Output("output_mention_indices: int32");

class ExtractMentionsCKYOp : public OpKernel {
public:
  explicit ExtractMentionsCKYOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mentions_per_word", &_mentions_per_word));
  }

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstVec mention_scores = context->input(0).vec<float>();
    TTypes<int32>::ConstVec candidate_starts = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec candidate_ends = context->input(2).vec<int32>();
    TTypes<int32>::ConstVec sentence_lengths = context->input(3).vec<int32>();

    int num_input_mentions = mention_scores.dimension(0);
    int num_sentences = sentence_lengths.dimension(0);

    std::map<std::pair<int, int>, int> mention_map;
    for (int i = 0; i < num_input_mentions; ++i) {
      mention_map[{candidate_starts(i), candidate_ends(i)}] = i;
    }
    std::vector<int> top_spans;

    int sentence_offset = 0;
    for (int i = 0; i < num_sentences; ++i) {
      get_viterbi_spans(sentence_offset, sentence_lengths(i), mention_map, mention_scores, candidate_starts, candidate_ends, &top_spans);
      sentence_offset += sentence_lengths(i);
    }

    Tensor* output_mention_indices_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {static_cast<int64>(top_spans.size())}, &output_mention_indices_tensor));
    TTypes<int32>::Vec output_mention_indices = output_mention_indices_tensor->vec<int32>();
    for (int i = 0; i < top_spans.size(); ++i) {
      output_mention_indices(i) = top_spans[i];
    }
  }
private:
  float _mentions_per_word;

  struct Cell {
    float score;
    int index;
    std::vector<const Cell *> top_cells;
  };

  void get_viterbi_spans(int sentence_offset,
                         int sentence_length,
                         const std::map<std::pair<int, int>, int> &mention_map,
                         const TTypes<float>::ConstVec &mention_scores,
                         const TTypes<int32>::ConstVec &candidate_starts,
                         const TTypes<int32>::ConstVec &candidate_ends,
                         std::vector<int> *top_spans) {
    int max_mentions = floor(sentence_length * _mentions_per_word);
    Cell chart[sentence_length][sentence_length];

    for (int width = 0; width < sentence_length; ++width) { // Iterate over span widths (minus one).
      for (int i = 0; i < sentence_length - width; ++i) { // Iterate over span starts.
        int j = i + width; // Span end.
        Cell &current_cell = chart[i][j];
        auto it = mention_map.find({sentence_offset + i, sentence_offset + j});
        if (it != mention_map.end()) {
          current_cell.index = it->second;
          current_cell.score = mention_scores(current_cell.index);
        } else {
          current_cell.index = -1;
          current_cell.score = -std::numeric_limits<float>::infinity(); // This should never be accessed.
        }
        if (i == j) {
          current_cell.top_cells.push_back(&current_cell);
        } else {
          float best_split_score = -std::numeric_limits<float>::infinity();
          int best_split = -1;
          for (int k = i; k < j; ++k) { // Iterate over span splits.
            float merged_sum = 0;
            merge_top_mentions(max_mentions, current_cell, chart[i][k], chart[k+1][j], [&merged_sum](const Cell &cell) {
                merged_sum += cell.score;
              });
            if (merged_sum > best_split_score) {
              best_split_score = merged_sum;
              best_split = k;
            }
          }
          CHECK_GE(best_split, 0);
          merge_top_mentions(max_mentions, current_cell, chart[i][best_split], chart[best_split+1][j], [&current_cell](const Cell &cell) {
              current_cell.top_cells.push_back(&cell);
            });
        }
      }
    }

    std::vector<const Cell *> &top_cells = chart[0][sentence_length-1].top_cells;
    std::sort(top_cells.begin(), top_cells.end(),
              [&candidate_starts, &candidate_ends] (const Cell *c1, const Cell *c2) {
                int i1 = c1->index;
                int i2 = c2->index;
                if (candidate_starts(i1) < candidate_starts(i2)) {
                  return true;
                } else if (candidate_starts(i1) > candidate_starts(i2)) {
                  return false;
                } else if (candidate_ends(i1) < candidate_ends(i2)) {
                  return true;
                } else if (candidate_ends(i1) > candidate_ends(i2)) {
                  return false;
                } else {
                  return i1 < i2;
                }
              });
    for (auto cell : top_cells) {
      top_spans->push_back(cell->index);
    }
  }

  void merge_top_mentions(int max_mentions, const Cell& current_cell, const Cell &left_cell, const Cell &right_cell, std::function<void (const Cell&)> callback) {
    int left_idx = 0;
    int right_idx = 0;
    bool consider_current = current_cell.index >= 0; // Don't consider the current cell if it is not in the candidate set.

    int callback_count = 0;

    // 3-way merge between current, left, and right.
    while (callback_count < max_mentions && left_idx < left_cell.top_cells.size() && right_idx < right_cell.top_cells.size() && consider_current) {
      const Cell &top_left = *left_cell.top_cells[left_idx];
      const Cell &top_right = *right_cell.top_cells[right_idx];
      if (top_left.score > top_right.score) {
        if (top_left.score > current_cell.score) {
          callback(top_left);
          ++left_idx;
        } else {
          callback(current_cell);
          consider_current = false;
        }
      } else {
        if (top_right.score > current_cell.score) {
          callback(top_right);
          ++right_idx;
        } else {
          callback(current_cell);
          consider_current = false;
        }
      }
      ++callback_count;
    }

    // 2-way merge between left and right.
    while (callback_count < max_mentions && left_idx < left_cell.top_cells.size() && right_idx < right_cell.top_cells.size()) {
      const Cell &top_left = *left_cell.top_cells[left_idx];
      const Cell &top_right = *right_cell.top_cells[right_idx];
      if (top_left.score > top_right.score) {
        callback(top_left);
        ++left_idx;
      } else {
        callback(top_right);
        ++right_idx;
      }
      ++callback_count;
    }

    // 2-way merge between current and left.
    while (callback_count < max_mentions && left_idx < left_cell.top_cells.size() && consider_current) {
      const Cell &top_left = *left_cell.top_cells[left_idx];
      if (top_left.score > current_cell.score) {
        callback(top_left);
        ++left_idx;
      } else {
        callback(current_cell);
        consider_current = false;
      }
      ++callback_count;
    }

    // 2-way merge between current and right.
    while (callback_count < max_mentions && right_idx < right_cell.top_cells.size() && consider_current) {
      const Cell &top_right = *right_cell.top_cells[right_idx];
      if (top_right.score > current_cell.score) {
        callback(top_right);
        ++right_idx;
      } else {
        callback(current_cell);
        consider_current = false;
      }
      ++callback_count;
    }

    // Copy remaining values.
    while (callback_count < max_mentions && left_idx < left_cell.top_cells.size()) {
      const Cell &top_left = *left_cell.top_cells[left_idx];
      callback(top_left);
      ++left_idx;
      ++callback_count;
    }
    while (callback_count < max_mentions && right_idx < right_cell.top_cells.size()) {
      const Cell &top_right = *right_cell.top_cells[right_idx];
      callback(top_right);
      ++right_idx;
      ++callback_count;
    }
    if (callback_count < max_mentions && consider_current) {
      callback(current_cell);
      consider_current = false;
      ++callback_count;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ExtractMentionsCKY").Device(DEVICE_CPU), ExtractMentionsCKYOp);
