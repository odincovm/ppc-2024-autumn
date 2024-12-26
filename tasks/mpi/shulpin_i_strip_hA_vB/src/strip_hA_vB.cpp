#include "mpi/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

void shulpin_strip_scheme_A_B::calculate_mpi(int rows_a, int cols_a, int cols_b, std::vector<int> A_mpi,
                                             std::vector<int> B_mpi, std::vector<int>& C_mpi) {
  boost::mpi::communicator world;
  int ProcRank = world.rank();
  int ProcNum = world.size();

  int ProcPartRows = rows_a / ProcNum;
  int RemainingRows = rows_a % ProcNum;
  int LocalRows = (ProcRank < RemainingRows) ? ProcPartRows + 1 : ProcPartRows;

  std::vector<int> bufA(LocalRows * cols_a, 0);
  std::vector<int> bufB;
  std::vector<int> bufC(LocalRows * cols_b, 0);

  std::vector<int> sendcounts(ProcNum);
  std::vector<int> displs(ProcNum);

  if (ProcRank == 0) {
    for (int i = 0; i < ProcNum; ++i) {
      sendcounts[i] = ((i < RemainingRows) ? (ProcPartRows + 1) : ProcPartRows) * cols_a;
      displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }
  }

  std::vector<boost::mpi::request> requests;
  if (ProcRank == 0) {
    for (int i = 0; i < ProcNum; ++i) {
      int offset = displs[i];
      int count = sendcounts[i];
      if (i == 0) {
        std::copy(A_mpi.begin(), A_mpi.begin() + count, bufA.begin());
      } else {
        requests.push_back(world.isend(i, 0, &A_mpi[offset], count));
      }
    }
  } else {
    requests.push_back(world.irecv(0, 0, bufA.data(), LocalRows * cols_a));
  }

  boost::mpi::wait_all(requests.begin(), requests.end());

  if (ProcRank == 0) {
    bufB = std::move(B_mpi);
  }

  requests.clear();
  if (ProcRank == 0) {
    for (int i = 1; i < ProcNum; ++i) {
      requests.push_back(world.isend(i, 1, bufB.data(), bufB.size()));
    }
  } else {
    bufB.resize(cols_a * cols_b);
    requests.push_back(world.irecv(0, 1, bufB.data(), bufB.size()));
  }

  boost::mpi::wait_all(requests.begin(), requests.end());

  std::fill(bufC.begin(), bufC.end(), 0);

  for (int i = 0; i < LocalRows; ++i) {
    for (int k = 0; k < cols_a; ++k) {
      int temp = bufA[i * cols_a + k];
      const int* B_row = &bufB[k * cols_b];
      int* C_row = &bufC[i * cols_b];
      for (int j = 0; j < cols_b; ++j) {
        C_row[j] += temp * B_row[j];
      }
    }
  }

  if (ProcRank == 0) {
    C_mpi.resize(rows_a * cols_b, 0);
    for (int i = 0; i < ProcNum; ++i) {
      sendcounts[i] = ((i < RemainingRows) ? (ProcPartRows + 1) : ProcPartRows) * cols_b;
      if (i > 0) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
      }
    }
  }

  requests.clear();
  if (ProcRank == 0) {
    for (int i = 1; i < ProcNum; ++i) {
      requests.push_back(world.irecv(i, 2, &C_mpi[displs[i]], sendcounts[i]));
    }
    std::copy(bufC.begin(), bufC.end(), C_mpi.begin());
  } else {
    requests.push_back(world.isend(0, 2, bufC.data(), LocalRows * cols_b));
  }

  boost::mpi::wait_all(requests.begin(), requests.end());
}

void shulpin_strip_scheme_A_B::calculate_seq(int rows_a, int cols_a, int cols_b, std::vector<int> A_seq,
                                             std::vector<int> B_seq, std::vector<int>& C_seq) {
  int* A_ptr = A_seq.data();
  int* B_ptr = B_seq.data();
  int* C_ptr = C_seq.data();

  for (int i = 0; i < rows_a; ++i) {
    int* C_row_ptr = C_ptr + i * cols_b;
    for (int k = 0; k < cols_a; ++k) {
      int a_val = A_ptr[i * cols_a + k];
      const int* B_col_ptr = B_ptr + k * cols_b;
      for (int j = 0; j < cols_b; ++j) {
        C_row_ptr[j] += a_val * B_col_ptr[j];
      }
    }
  }
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_par::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    mpi_cols_A = *reinterpret_cast<int*>(taskData->inputs[2]);
    mpi_rows_A = *reinterpret_cast<int*>(taskData->inputs[3]);
    mpi_cols_B = *reinterpret_cast<int*>(taskData->inputs[4]);
    mpi_rows_B = *reinterpret_cast<int*>(taskData->inputs[5]);

    mpi_A.assign(reinterpret_cast<int*>(taskData->inputs[0]),
                 reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    mpi_B.assign(reinterpret_cast<int*>(taskData->inputs[1]),
                 reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1]);

    mpi_result.resize(taskData->outputs_count[0], 0);
  }
  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_par::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int a_cols = *reinterpret_cast<int*>(taskData->inputs[2]);
    int a_rows = *reinterpret_cast<int*>(taskData->inputs[3]);
    int b_cols = *reinterpret_cast<int*>(taskData->inputs[4]);
    int b_rows = *reinterpret_cast<int*>(taskData->inputs[5]);
    return (taskData->inputs_count.size() > 4 && !taskData->outputs_count.empty() &&
            (a_cols > 0 && a_rows > 0 && b_cols > 0 && b_rows > 0) && (a_cols == b_rows));
  }
  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_par::run() {
  internal_order_test();

  std::vector<int> meta_data = {mpi_cols_A, mpi_rows_A, mpi_cols_B, mpi_rows_B};
  boost::mpi::broadcast(world, meta_data, 0);

  if (world.rank() != 0) {
    mpi_cols_A = meta_data[0];
    mpi_rows_A = meta_data[1];
    mpi_cols_B = meta_data[2];
    mpi_rows_B = meta_data[3];
  }

  std::vector<int> local_res(mpi_rows_A * mpi_cols_B, 0);
  calculate_mpi(mpi_rows_A, mpi_cols_A, mpi_cols_B, mpi_A, mpi_B, local_res);

  boost::mpi::reduce(world, local_res, mpi_result, std::plus<>(), 0);
  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_par::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(mpi_result.begin(), mpi_result.end(), output);
  }

  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::pre_processing() {
  internal_order_test();

  int cols_A_tmp = *reinterpret_cast<int*>(taskData->inputs[2]);
  int rows_A_tmp = *reinterpret_cast<int*>(taskData->inputs[3]);

  seq_cols_A = cols_A_tmp;
  seq_rows_A = rows_A_tmp;

  int cols_B_tmp = *reinterpret_cast<int*>(taskData->inputs[4]);
  int rows_B_tmp = *reinterpret_cast<int*>(taskData->inputs[5]);

  seq_cols_B = cols_B_tmp;
  seq_rows_B = rows_B_tmp;

  std::vector<int> A_tmp{};
  std::vector<int> B_tmp{};

  int* A_tmp_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int A_tmp_size = taskData->inputs_count[0];
  A_tmp.assign(A_tmp_data, A_tmp_data + A_tmp_size);

  int* B_tmp_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int B_tmp_size = taskData->inputs_count[1];
  B_tmp.assign(B_tmp_data, B_tmp_data + B_tmp_size);

  seq_A = A_tmp;
  seq_B = B_tmp;

  int res_size = taskData->outputs_count[0];
  seq_result.resize(res_size, 0);

  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::validation() {
  internal_order_test();

  int a_cols = *reinterpret_cast<int*>(taskData->inputs[2]);
  int a_rows = *reinterpret_cast<int*>(taskData->inputs[3]);
  int b_cols = *reinterpret_cast<int*>(taskData->inputs[4]);
  int b_rows = *reinterpret_cast<int*>(taskData->inputs[5]);

  return (taskData->inputs_count.size() > 4 && !taskData->outputs_count.empty() &&
          (a_cols > 0 && a_rows > 0 && b_cols > 0 && b_rows > 0) && (a_cols == b_rows));
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::run() {
  internal_order_test();
  calculate_seq(seq_rows_A, seq_cols_A, seq_cols_B, seq_A, seq_B, seq_result);
  return true;
}

bool shulpin_strip_scheme_A_B::Matrix_hA_vB_seq::post_processing() {
  internal_order_test();

  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(seq_result.begin(), seq_result.end(), output);

  return true;
}