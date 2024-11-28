
#include "seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp"

using namespace std::chrono_literals;

bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::validation() {
  internal_order_test();

  if ((taskData->inputs_count[0] == 0) || (taskData->inputs_count[2] == 0) || (taskData->outputs_count[0] == 0))

    return false;

  if (taskData->inputs_count[1] != (taskData->inputs_count[2] / taskData->inputs_count[3])) return false;

  if (((taskData->inputs_count[0] % taskData->inputs_count[1]) != 0) ||
      ((taskData->inputs_count[2] % taskData->inputs_count[3]) != 0) ||
      (taskData->outputs_count[0] % taskData->outputs_count[1] != 0))
    return false;

  return true;
}

bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::pre_processing() {
  internal_order_test();

  szA.push_back(taskData->inputs_count[0]);

  szA.push_back(taskData->inputs_count[1]);
  szA.push_back(szA[0] / szA[1]);
  szB.push_back(taskData->inputs_count[2]);

  szB.push_back(taskData->inputs_count[3]);

  szB.push_back(szB[0] / szB[1]);
  szC.push_back(taskData->outputs_count[0]);
  szC.push_back(taskData->outputs_count[1]);
  szC.push_back(szC[0] / szC[1]);

  matrixA.assign(reinterpret_cast<double*>(taskData->inputs[0]),
                 reinterpret_cast<double*>(taskData->inputs[0]) + szA[0]);
  matrixB.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                 reinterpret_cast<double*>(taskData->inputs[1]) + szB[0]);

  matrixC.resize(szC[0]);
  for (int i = 0; i < szC[0]; i++) {
    matrixC[i] = 0;
  }
  return true;
}
bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::run() {
  internal_order_test();
  std::vector<double> ribbon(szB[1], 0);

  for (int i = 0; i < szB[2]; i++) {
    for (int j = 0; j < szB[1]; j++) {
      ribbon[j] = matrixB[szB[2] * j + i];
    }

    for (int Arow = 0; Arow < szA[1]; Arow++) {
      for (int k = 0; k < szB[1]; k++) {
        matrixC[Arow * szC[1] + i] += matrixA[Arow * szA[2] + k] * ribbon[k];
      }
    }
  }
  return true;
}
bool Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < szC[0]; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = matrixC[i];
  }
  return true;
}
