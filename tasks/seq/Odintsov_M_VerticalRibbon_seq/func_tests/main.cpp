
#include <gtest/gtest.h>

#include <seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp>
#include <vector>

TEST(Sequential_multy, sz_10000) {
  // Create data
  std::vector<double> matrixA(10000, 1);
  std::vector<double> matrixB(10000, 1);

  std::vector<double> matrixC(10000, 100);
  std::vector<double> out(matrixC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(10000);
  taskDataSeq->inputs_count.emplace_back(100);
  taskDataSeq->inputs_count.emplace_back(10000);
  taskDataSeq->inputs_count.emplace_back(100);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(10000);
  taskDataSeq->outputs_count.emplace_back(100);

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_EQ(testClass.validation(), true);

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  for (size_t i = 0; i < matrixC.size(); i++) ASSERT_EQ(matrixC[i], out[i]);
}
TEST(Sequential_multy, sz_40000) {
  // Create data
  std::vector<double> matrixA(40000, 1);
  std::vector<double> matrixB(40000, 1);

  std::vector<double> matrixC(40000, 200);
  std::vector<double> out(matrixC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(40000);
  taskDataSeq->inputs_count.emplace_back(200);
  taskDataSeq->inputs_count.emplace_back(40000);
  taskDataSeq->inputs_count.emplace_back(200);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(40000);
  taskDataSeq->outputs_count.emplace_back(200);

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_EQ(testClass.validation(), true);

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  for (size_t i = 0; i < matrixC.size(); i++) ASSERT_EQ(matrixC[i], out[i]);
}
TEST(Sequential_multy, dfsz_1200) {
  // Увеличить размер матрицы
  // Create data
  std::vector<double> matrixA(1200, 1);
  std::vector<double> matrixB(1200, 1);

  std::vector<double> matrixC(900, 40);
  std::vector<double> out(matrixC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(1200);
  taskDataSeq->inputs_count.emplace_back(30);
  taskDataSeq->inputs_count.emplace_back(1200);
  taskDataSeq->inputs_count.emplace_back(40);
  ;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(900);
  taskDataSeq->outputs_count.emplace_back(30);

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_EQ(testClass.validation(), true);

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  for (size_t i = 0; i < matrixC.size(); i++) ASSERT_EQ(matrixC[i], out[i]);
}
