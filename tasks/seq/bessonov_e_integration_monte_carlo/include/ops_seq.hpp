#pragma once

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_integration_monte_carlo_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a, b;
  int num_points;
  std::function<double(double)> exampl_func;

 private:
  double res{};
};

}  // namespace bessonov_e_integration_monte_carlo_seq