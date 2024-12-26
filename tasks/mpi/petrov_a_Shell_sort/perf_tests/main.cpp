#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

TEST(petrov_a_Shell_sort_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int vector_size = 10000;
  std::vector<int> data(vector_size);
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    int current = vector_size;
    std::generate(data.begin(), data.end(), [&current]() { return current--; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<petrov_a_Shell_sort_mpi::TestTaskMPI>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

TEST(petrov_a_Shell_sort_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int vector_size = 10000;
  std::vector<int> data(vector_size);
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    int current = vector_size;
    std::generate(data.begin(), data.end(), [&current]() { return current--; });
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<petrov_a_Shell_sort_mpi::TestTaskMPI>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}
