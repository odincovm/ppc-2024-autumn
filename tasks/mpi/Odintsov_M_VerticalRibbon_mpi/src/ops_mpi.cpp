#include "mpi/Odintsov_M_VerticalRibbon_mpi/include/ops_mpi.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
using namespace std::chrono_literals;

namespace Odintsov_M_VerticalRibbon_mpi {
// Последовательная версия
bool VerticalRibbonMPISequential::validation() {
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
bool VerticalRibbonMPISequential::pre_processing() {
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

  matrixA.assign(reinterpret_cast<double *>(taskData->inputs[0]),
                 reinterpret_cast<double *>(taskData->inputs[0]) + szA[0]);
  matrixB.assign(reinterpret_cast<double *>(taskData->inputs[1]),
                 reinterpret_cast<double *>(taskData->inputs[1]) + szB[0]);

  matrixC.resize(szC[0]);
  for (int i = 0; i < szC[0]; i++) {
    matrixC[i] = 0;
  }
  return true;
}
bool VerticalRibbonMPISequential::run() {
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
bool VerticalRibbonMPISequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < szC[0]; i++) {
    reinterpret_cast<double *>(taskData->outputs[0])[i] = matrixC[i];
  }
  return true;
}

bool VerticalRibbonMPIParallel::validation() {
  internal_order_test();

  if (com.rank() == 0) {
    if ((taskData->inputs_count[0] == 0) || (taskData->inputs_count[2] == 0) || (taskData->outputs_count[0] == 0))
      return false;

    if (taskData->inputs_count[1] != (taskData->inputs_count[2] / taskData->inputs_count[3])) return false;

    if (((taskData->inputs_count[0] % taskData->inputs_count[1]) != 0) ||
        ((taskData->inputs_count[2] % taskData->inputs_count[3]) != 0) ||
        (taskData->outputs_count[0] % taskData->outputs_count[1] != 0))
      return false;
  }
  return true;
}

bool VerticalRibbonMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    szA.push_back(taskData->inputs_count[0]);

    szA.push_back(taskData->inputs_count[1]);
    szA.push_back(szA[0] / szA[1]);
    szB.push_back(taskData->inputs_count[2]);

    szB.push_back(taskData->inputs_count[3]);

    szB.push_back(szB[0] / szB[1]);
    szC.push_back(taskData->outputs_count[0]);
    szC.push_back(taskData->outputs_count[1]);
    szC.push_back(szC[0] / szC[1]);

    matrixA.assign(reinterpret_cast<double *>(taskData->inputs[0]),
                   reinterpret_cast<double *>(taskData->inputs[0]) + szA[0]);
    matrixB.assign(reinterpret_cast<double *>(taskData->inputs[1]),
                   reinterpret_cast<double *>(taskData->inputs[1]) + szB[0]);

    matrixC.resize(szC[0]);
    for (int i = 0; i < szC[0]; i++) {
      matrixC[i] = 0;
    }
    ribbon_sz = 0;
  }
  return true;
}
bool VerticalRibbonMPIParallel::run() {
  internal_order_test();

  if (com.rank() != 0) {
    szA.resize(3);
    szB.resize(3);
  }
  for (int i = 0; i < 3; i++) {
    broadcast(com, szA[i], 0);
    broadcast(com, szB[i], 0);
  }

  if (com.rank() != 0) {
    matrixA.resize(szA[0]);
  }
  for (int i = 0; i < szA[0]; i++) {
    broadcast(com, matrixA[i], 0);
  }

  if (com.rank() == 0) {
    ribbon_sz = (szB[2] + com.size() - 1) / com.size();
  }

  broadcast(com, ribbon_sz, 0);

  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      std::vector<double> ribbon;

      for (int j = 0; j < szB[1]; j++) {
        int startcol = pr * ribbon_sz;
        int endcol = (pr + 1) * ribbon_sz;

        for (int i = startcol; i < endcol; i++) {
          ribbon.push_back(matrixB[szB[2] * j + i]);
        }
      }

      com.send(pr, 0, ribbon.data(), ribbon.size());
    }
  }

  if (com.rank() == 0) {
    for (int j = 0; j < szB[1]; j++) {
      for (int i = 0; i < ribbon_sz; i++) {
        local_ribbon.push_back(matrixB[szB[2] * j + i]);
      }
    }
  } else {
    std::vector<double> buffer(ribbon_sz * szB[1], 0);
    com.recv(0, 0, buffer.data(), buffer.size());
    local_ribbon.insert(local_ribbon.end(), buffer.begin(), buffer.end());
  }

  for (int Bcol = 0; Bcol < ribbon_sz; Bcol++) {
    for (int Arow = 0; Arow < szA[1]; Arow++) {
      double sum = 0;

      for (int k = 0; k < szB[1]; k++) {
        sum += matrixA[Arow * szA[2] + k] * local_ribbon[k * ribbon_sz + Bcol];
      }
      local_mC.push_back(sum);
    }
  }

  gather(com, local_mC.data(), local_mC.size(), matrixC, 0);

  if (com.rank() == 0) {
    std::vector<double> transposed(matrixC.size());

    for (int i = 0; i < szC[1]; ++i) {
      for (int j = 0; j < szC[2]; ++j) {
        transposed[j * szC[1] + i] = matrixC[i * szC[2] + j];
      }
    }
    matrixC = std::move(transposed);
  }

  return true;
}

bool VerticalRibbonMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    for (int i = 0; i < szC[0]; i++) {
      reinterpret_cast<double *>(taskData->outputs[0])[i] = matrixC[i];
    }
  }
  return true;
}
};  // namespace Odintsov_M_VerticalRibbon_mpi
