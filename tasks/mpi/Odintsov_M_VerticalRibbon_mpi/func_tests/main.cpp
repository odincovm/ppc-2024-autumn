
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Odintsov_M_VerticalRibbon_mpi/include/ops_mpi.hpp"

TEST(Parallel_MPI_matrix, sz_3600) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(3600, 1);
  std::vector<double> matrixB(3600, 1);
  std::vector<double> out(3600, 0);
  std::vector<double> out_s(3600, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataPar->inputs_count.emplace_back(3600);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->inputs_count.emplace_back(3600);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(3600);
    taskDataPar->outputs_count.emplace_back(60);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(3600);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->inputs_count.emplace_back(3600);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(3600);
    taskDataSeq->outputs_count.emplace_back(60);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out_s[i]);
    }
  }
}

TEST(Parallel_MPI_matrix, sz_90000) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(90000, 1);
  std::vector<double> matrixB(90000, 1);
  std::vector<double> out(90000, 0);
  std::vector<double> out_s(90000, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataPar->inputs_count.emplace_back(90000);
    taskDataPar->inputs_count.emplace_back(300);
    taskDataPar->inputs_count.emplace_back(90000);
    taskDataPar->inputs_count.emplace_back(300);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(90000);
    taskDataPar->outputs_count.emplace_back(300);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(90000);
    taskDataSeq->inputs_count.emplace_back(300);
    taskDataSeq->inputs_count.emplace_back(90000);
    taskDataSeq->inputs_count.emplace_back(300);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(90000);
    taskDataSeq->outputs_count.emplace_back(300);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out_s[i]);
    }
  }
}

TEST(Parallel_MPI_matrix, difsz_1800) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(1800, 1);
  std::vector<double> matrixB(1800, 1);
  std::vector<double> out(3600, 0);
  std::vector<double> out_s(3600, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataPar->inputs_count.emplace_back(1800);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->inputs_count.emplace_back(1800);
    taskDataPar->inputs_count.emplace_back(30);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(3600);
    taskDataPar->outputs_count.emplace_back(60);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(1800);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->inputs_count.emplace_back(1800);
    taskDataSeq->inputs_count.emplace_back(30);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(3600);
    taskDataSeq->outputs_count.emplace_back(60);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out_s[i]);
    }
  }
}
