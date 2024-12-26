// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/example/include/ops_mpi.hpp"

TEST(mpi_example_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<nesterov_a_test_task_mpi::TestMPITaskParallel>(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}

TEST(mpi_example_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 120;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<nesterov_a_test_task_mpi::TestMPITaskParallel>(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(count_size_vector, global_sum[0]);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0 && (argc < 2 || argv[1] != std::string("--full-workers-log"))) {
    class WorkersTestPrinter : public ::testing::EmptyTestEventListener {
     public:
      WorkersTestPrinter(std::unique_ptr<TestEventListener>&& base, int rank) : base_(std::move(base)), rank_(rank) {}

      void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (test_info.result()->Passed()) {
          return;
        }
        print_process_rank();
        base_->OnTestEnd(test_info);
      }

      void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        print_process_rank();
        base_->OnTestPartResult(test_part_result);
      }

     private:
      void print_process_rank() const { printf(" [  PROCESS %d  ] ", rank_); }

      std::unique_ptr<TestEventListener> base_;
      int rank_;
    };
    listeners.Append(new WorkersTestPrinter(
        std::unique_ptr<::testing::TestEventListener>(listeners.Release(listeners.default_result_printer())),
        world.rank()));
  }
  struct BufferGarbageDetector : public ::testing::EmptyTestEventListener {
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
      world.barrier();
      if (const auto status = world.iprobe(boost::mpi::any_source, boost::mpi::any_tag)) {
        fprintf(stderr, "[  PROCESS %d  ] [  FAILED  ] %s.%s: MPI buffer is cluttered, unread message tag is %d\n",
                world.rank(), test_info.test_suite_name(), test_info.name(), status->tag());
        exit(2);
      }
      world.barrier();
    }

    boost::mpi::communicator world;
  };
  listeners.Append(new BufferGarbageDetector);
  return RUN_ALL_TESTS();
}
