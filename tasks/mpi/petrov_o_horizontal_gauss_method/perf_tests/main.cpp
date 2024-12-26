#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"

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

template <typename TaskType>
void runTaskTest(size_t n, int num_running) {
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output(n);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    generateRandomMatrixAndB(n, input_matrix, input_b);

    taskData->inputs_count.emplace_back(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.emplace_back(n * sizeof(double));
  }

  auto task = std::make_shared<TaskType>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_running;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);

  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < n; ++j) {
        ax_i += input_matrix[i * n + j] * output[j];
      }
      residual += std::pow(ax_i - input_b[i], 2);
    }
    residual = std::sqrt(residual);
    ASSERT_NEAR(residual, 0.0, 1e-10);
  }
}

template <typename TaskType>
void runPipelineTest(size_t n, int num_running) {
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output(n);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    generateRandomMatrixAndB(n, input_matrix, input_b);

    taskData->inputs_count.emplace_back(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.emplace_back(n * sizeof(double));
  }
  auto task = std::make_shared<TaskType>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_running;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < n; ++j) {
        ax_i += input_matrix[i * n + j] * output[j];
      }
      residual += std::pow(ax_i - input_b[i], 2);
    }
    residual = std::sqrt(residual);
    ASSERT_NEAR(residual, 0.0, 1e-10);
  }
}

TEST(petrov_o_horizontal_gauss_method_seq, test_pipeline_run) {
  boost::mpi::communicator world;
  runPipelineTest<petrov_o_horizontal_gauss_method_mpi::SequentialTask>(10, 100);
}

TEST(petrov_o_horizontal_gauss_method_seq, test_task_run) {
  boost::mpi::communicator world;
  runTaskTest<petrov_o_horizontal_gauss_method_mpi::SequentialTask>(10, 100);
}

TEST(petrov_o_horizontal_gauss_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  runPipelineTest<petrov_o_horizontal_gauss_method_mpi::ParallelTask>(10, 100);
}

TEST(petrov_o_horizontal_gauss_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  runTaskTest<petrov_o_horizontal_gauss_method_mpi::ParallelTask>(10, 100);
}