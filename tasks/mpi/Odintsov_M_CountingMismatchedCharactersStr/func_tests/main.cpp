
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

TEST(Parallel_MPI_count, ans_10) {
  // Create data//
  boost::mpi::communicator com;

  char* str1 = new char[7];
  char* str2 = new char[7];
  strcpy_s(str1, 7, "qwertp");
  strcpy_s(str2, 7, "qellow");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  ASSERT_EQ(10, out[0]);
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(10, out[0]);
  }
}

TEST(Parallel_MPI_count, ans_6) {
  // Create data
  boost::mpi::communicator com;

  char* str1 = new char[9];
  char* str2 = new char[9];
  strcpy_s(str1, 9, "lsjgjeqw");
  strcpy_s(str2, 9, "udjgjeqp");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel testClassPar(taskDataPar);
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  ASSERT_EQ(6, out[0]);
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPISequential testClassSeq(taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(6, out[0]);
  }
}