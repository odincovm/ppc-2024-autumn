
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Odintsov_M_VerticalRibbon_mpi {

class VerticalRibbonMPISequential : public ppc::core::Task {
 public:
  explicit VerticalRibbonMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std ::vector<double> matrixA;
  std::vector<double> matrixB;
  std::vector<double> matrixC;
  // [0] - размер, [1] - количество строк, [2] - количество столбцов
  std::vector<int> szA;
  std::vector<int> szB;
  std::vector<int> szC;
};

class VerticalRibbonMPIParallel : public ppc::core::Task {
 public:
  explicit VerticalRibbonMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> local_ribbon;
  std::vector<double> local_mC;
  int ribbon_sz;
  std ::vector<double> matrixA;
  std::vector<double> matrixB;
  std::vector<double> matrixC;
  // [0] - размер, [1] - количество строк, [2] - количество столбцов
  std::vector<int> szA;
  std::vector<int> szB;
  std::vector<int> szC;
  boost::mpi::communicator com;
};
}  // namespace Odintsov_M_VerticalRibbon_mpi