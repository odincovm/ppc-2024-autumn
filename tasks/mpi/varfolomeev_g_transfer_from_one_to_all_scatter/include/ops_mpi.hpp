// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_transfer_from_one_to_all_scatter_mpi {
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_values, local_input_values;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};
}  // namespace varfolomeev_g_transfer_from_one_to_all_scatter_mpi