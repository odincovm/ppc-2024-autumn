
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include "mpi/Odintsov_M_GlobalOptimizationSpecifications/include/ops_mpi.hpp"


static std::vector<double> createFunc(int min, int max) {
  std::vector<double> func;
  srand(time(nullptr));
  for (int i = 0; i < 2; i++) {
    func.push_back(min + rand() % (max - min + 1));
  }
  return func;
}


static std::vector<double> creareConstr(int min, int max, int count) {
  std::vector<double> constr;
  srand(time(nullptr));
  for (int i = 0; i < 3*count; i++) {
    constr.push_back(min + rand() % (max - min + 1));
  }
  return constr;
}

TEST(Odintsov_M_OptimPar_MPI, test_min_1) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10,10,36);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(36);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  
  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);
  
  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(36);  // ���������� �����������
    taskDataSeq->inputs_count.emplace_back(0);  // �����
  
    
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_min_2) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, 10, 24);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(24);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(24);  // ���������� �����������
    taskDataSeq->inputs_count.emplace_back(0);   // �����

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_max_1) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, 10, 12);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(12);
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(12);  // ���������� �����������
    taskDataSeq->inputs_count.emplace_back(1);   // �����

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}