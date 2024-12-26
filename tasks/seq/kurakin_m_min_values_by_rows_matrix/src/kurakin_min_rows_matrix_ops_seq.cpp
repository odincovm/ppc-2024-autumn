// Copyright 2024 Nesterov Alexander
#include "seq/kurakin_m_min_values_by_rows_matrix/include/kurakin_min_rows_matrix_ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>

bool kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  // Init value for output
  count_rows = (int)*taskData->inputs[1];
  size_rows = (int)*taskData->inputs[2];
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  res = std::vector<int>(count_rows, 0);

  for (int i = 0; i < count_rows; i++) {
    res[i] = *std::min_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}