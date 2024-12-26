// Copyright 2024 Korobeinikov Arseny
#include "mpi/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_mpi_korobeinikov.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> korobeinikov_a_test_task_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  input_.reserve(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], std::back_inserter(input_));

  count_rows = (int)*taskData->inputs[1];
  if (count_rows != 0) {
    size_rows = (int)(taskData->inputs_count[0] / (*taskData->inputs[1]));
  } else {
    size_rows = 0;
  }
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if ((*taskData->inputs[1]) == 0) {
    return true;
  }
  return (*taskData->inputs[1] == taskData->outputs_count[0] &&
          (taskData->inputs_count[0] % (*taskData->inputs[1])) == 0);
}

bool korobeinikov_a_test_task_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    res[i] = *std::max_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
  }
  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((*taskData->inputs[1]) == 0) {
      return true;
    }
    return (*taskData->inputs[1] == taskData->outputs_count[0] &&
            (taskData->inputs_count[0] % (*taskData->inputs[1])) == 0);
  }
  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;

  if (world.rank() == 0) {
    count_rows = (int)*taskData->inputs[1];
    if (count_rows != 0) {
      size_rows = (int)(taskData->inputs_count[0] / (*taskData->inputs[1]));
    } else {
      size_rows = 0;
    }
    if (count_rows != 0) {
      num_use_proc = std::min(world.size(), count_rows * size_rows);
    } else {
      num_use_proc = world.size();
    }
    delta = taskData->inputs_count[0] / num_use_proc;
  }
  broadcast(world, delta, 0);
  broadcast(world, count_rows, 0);
  if (count_rows == 0) {
    return true;
  }
  broadcast(world, size_rows, 0);
  broadcast(world, num_use_proc, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_.reserve(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], std::back_inserter(input_));

    for (int proc = 1; proc < num_use_proc - 1; proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
    if (num_use_proc != 1) {
      int proc = num_use_proc - 1;
      world.send(proc, 0, input_.data() + proc * delta, delta + taskData->inputs_count[0] % num_use_proc);
    }
  }

  if (world.rank() == 0) {
    local_input_ = std::vector<int>(delta);
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    if (world.rank() == num_use_proc - 1 && num_use_proc != 0) {
      local_input_ = std::vector<int>(delta + (count_rows * size_rows) % num_use_proc);
      world.recv(0, 0, local_input_.data(), delta + (count_rows * size_rows) % num_use_proc);
    } else {
      if (world.rank() < num_use_proc) {
        local_input_ = std::vector<int>(delta);
        world.recv(0, 0, local_input_.data(), delta);
      }
    }
  }

  res = std::vector<int>(count_rows, 0);

  size_t default_local_size = 0;
  if (world.rank() == 0) {
    default_local_size = local_input_.size();
  }
  broadcast(world, default_local_size, 0);

  if (world.rank() < num_use_proc) {
    unsigned int ind = (world.rank() * default_local_size) / size_rows;
    for (unsigned int i = 0; i < ind; ++i) {
      reduce(world, INT_MIN, res[i], boost::mpi::maximum<int>(), 0);
    }

    unsigned int near_end = std::min(local_input_.size(), size_rows - (world.rank() * default_local_size) % size_rows);
    int local_res;

    local_res = *std::max_element(local_input_.begin(), local_input_.begin() + near_end);
    reduce(world, local_res, res[ind], boost::mpi::maximum<int>(), 0);
    ++ind;

    unsigned int k = 0;
    while (local_input_.begin() + near_end + k * size_rows < local_input_.end()) {
      local_res =
          *std::max_element(local_input_.begin() + near_end + k * size_rows,
                            std::min(local_input_.end(), local_input_.begin() + near_end + (k + 1) * size_rows));
      reduce(world, local_res, res[ind], boost::mpi::maximum<int>(), 0);
      ++k;
      ++ind;
    }

    for (unsigned int i = ind; i < res.size(); ++i) {
      reduce(world, INT_MIN, res[i], boost::mpi::maximum<int>(), 0);
    }
  } else {
    for (unsigned int i = 0; i < res.size(); ++i) {
      reduce(world, INT_MIN, res[i], boost::mpi::maximum<int>(), 0);
    }
  }
  return true;
}

bool korobeinikov_a_test_task_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < count_rows; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}