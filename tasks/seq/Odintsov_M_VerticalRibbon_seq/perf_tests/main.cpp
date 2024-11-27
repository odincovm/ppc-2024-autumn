
#include <gtest/gtest.h>

#include <vector>
#include <seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp>
#include "core/perf/include/perf.hpp"


TEST(sequential_matrix_perf_test, matrix_test_pipeline_run) {
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
  auto testClass = std::make_shared<Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);
 
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
 
  ppc::core::Perf::print_perf_statistic(perfResults);
  
  for (size_t i = 0; i < matrixC.size(); i++) ASSERT_EQ(matrixC[i], out[i]);
}

TEST(sequential_my_perf_test, test_task_run) {
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
  auto testClass = std::make_shared<Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClass);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < matrixC.size(); i++) ASSERT_EQ(matrixC[i], out[i]);
}