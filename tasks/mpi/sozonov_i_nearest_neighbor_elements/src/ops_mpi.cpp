#include "mpi/sozonov_i_nearest_neighbor_elements/include/ops_mpi.hpp"

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res = 0;
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 2;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int min = INT_MAX;
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    if (std::abs(input_[i + 1] - input_[i]) < min) {
      min = std::abs(input_[i + 1] - input_[i]);
      res = i;
    }
  }
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = input_[res];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = input_[res + 1];
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    n = taskData->inputs_count[0] - 1;
  }
  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of input and output
    return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 2;
  }
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, n, 0);

  std::vector<int> cnt(world.size());

  int delta = n / world.size();
  if (n % world.size() != 0) {
    delta++;
  }
  if (world.rank() >= world.size() - world.size() * delta + n) {
    delta--;
  }

  boost::mpi::gather(world, delta, cnt.data(), 0);

  if (world.rank() == 0) {
    diff = std::vector<std::pair<int, int>>(taskData->inputs_count[0] - 1);
    for (size_t i = 0; i < input_.size() - 1; ++i) {
      diff[i] = {std::abs(input_[i + 1] - input_[i]), i};
    }
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, diff.data() + proc * cnt[proc - 1], cnt[proc]);
    }
  }

  local_input_ = std::vector<std::pair<int, int>>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<std::pair<int, int>>(diff.begin(), diff.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  std::pair<int, int> local_res(INT_MAX, 0);
  local_res = *std::min_element(local_input_.begin(), local_input_.end());
  reduce(world, local_res, res, boost::mpi::minimum<std::pair<int, int>>(), 0);
  return true;
}

bool sozonov_i_nearest_neighbor_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = input_[res.second];
    reinterpret_cast<int*>(taskData->outputs[0])[1] = input_[res.second + 1];
  }
  return true;
}
