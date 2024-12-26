#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/petrov_o_horizontal_gauss_method/include/ops_seq.hpp"

void generateRandomMatrixAndB(size_t n, std::vector<double>& matrix, std::vector<double>& b) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);

  matrix.resize(n * n);
  b.resize(n);
  for (size_t i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  for (size_t i = 0; i < n; ++i) {
    b[i] = dis(gen);
  }
}

TEST(petrov_o_horizontal_gauss_method_seq, test_pipeline_run) {
  const int n = 100;
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output(n);

  generateRandomMatrixAndB(n, input_matrix, input_b);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(
      std::make_shared<petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential>(taskData));
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(petrov_o_horizontal_gauss_method_seq, test_task_run) {
  const int n = 100;
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output(n);

  generateRandomMatrixAndB(n, input_matrix, input_b);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(
      std::make_shared<petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential>(taskData));
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}