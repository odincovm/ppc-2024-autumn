
#pragma once

#include <string>
#include <vector>


#include "core/task/include/task.hpp"
namespace Odintsov_M_VerticalRibbon_seq {

class VerticalRibbonSequential : public ppc::core::Task {
 public:
  explicit VerticalRibbonSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  
  std ::vector<double> matrixA;
  std::vector<double> matrixB;
  std:: vector<double> matrixC;
  // [0] - размер, [1] - количество строк, [2] - количество столбцов
  std::vector<int> szA;
  std::vector<int> szB;
  std::vector<int> szC;
};

}  // namespace Odintsov_M_VerticalRibbon_seq