#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>

#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"

TEST(petrov_o_horizontal_gauss_method_par, TestGauss_Simple) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t n = 3;
  std::vector<double> input_matrix = {2, 1, 0, -3, -1, 2, 0, 1, 2};
  std::vector<double> input_b = {8, -11, -3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.emplace_back(n * sizeof(double));
  }

  petrov_o_horizontal_gauss_method_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    ASSERT_DOUBLE_EQ(output[0], 8);
    ASSERT_DOUBLE_EQ(output[1], -8);
    ASSERT_DOUBLE_EQ(output[2], 2.5);
  }
}

TEST(petrov_o_horizontal_gauss_method_par, TestGauss_RandomMatrix) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t n = 10;

  std::vector<double> random_matrix(n * n);
  std::vector<double> random_b(n);
  std::vector<double> par_output(n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100, 100);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      random_matrix[i * n + j] = dist(gen);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    random_b[i] = dist(gen);
  }

  std::shared_ptr<ppc::core::TaskData> par_taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_taskData->inputs_count.emplace_back(n);
    par_taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_matrix.data()));
    par_taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_b.data()));
    par_taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_output.data()));
    par_taskData->outputs_count.emplace_back(n * sizeof(double));
  }

  petrov_o_horizontal_gauss_method_mpi::ParallelTask par_task(par_taskData);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < n; ++j) {
        ax_i += random_matrix[i * n + j] * par_output[j];
      }
      residual += std::pow(ax_i - random_b[i], 2);
    }
    residual = std::sqrt(residual);
    ASSERT_LE(residual, 1e-8 * n * n * 100);
  }
}

TEST(petrov_o_horizontal_gauss_method_par, TestGauss_RandomMatrix_17x17) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t n = 17;

  std::vector<double> random_matrix(n * n);
  std::vector<double> random_b(n);
  std::vector<double> par_output(n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100, 100);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      random_matrix[i * n + j] = dist(gen);
    }
  }

  for (size_t i = 0; i < n; ++i) {
    random_b[i] = dist(gen);
  }

  std::shared_ptr<ppc::core::TaskData> par_taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_taskData->inputs_count.emplace_back(n);
    par_taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_matrix.data()));
    par_taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_b.data()));
    par_taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_output.data()));
    par_taskData->outputs_count.emplace_back(n * sizeof(double));
  }

  petrov_o_horizontal_gauss_method_mpi::ParallelTask par_task(par_taskData);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < n; ++j) {
        ax_i += random_matrix[i * n + j] * par_output[j];
      }
      residual += std::pow(ax_i - random_b[i], 2);
    }
    residual = std::sqrt(residual);
    ASSERT_LE(residual, 1e-8 * n * n * 100);
  }
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_Simple) {
  size_t n = 3;
  std::vector<double> input_matrix = {2, 1, 0, -3, -1, 2, 0, 1, 2};
  std::vector<double> input_b = {8, -11, -3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], 8);
  ASSERT_DOUBLE_EQ(output[1], -8);
  ASSERT_DOUBLE_EQ(output[2], 2.5);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_IdentityMatrix) {
  size_t n = 3;
  std::vector<double> input_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> input_b = {1, 2, 3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], 1);
  ASSERT_DOUBLE_EQ(output[1], 2);
  ASSERT_DOUBLE_EQ(output[2], 3);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_NegativeValues) {
  size_t n = 3;
  std::vector<double> input_matrix = {-2, -1, -1, -1, -1, -1, -1, -2, -1};
  std::vector<double> input_b = {-1, -2, -3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], -1);
  ASSERT_DOUBLE_EQ(output[1], 1);
  ASSERT_DOUBLE_EQ(output[2], 2);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_EmptyMatrix) {
  size_t n = 0;
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_mpi::SequentialTask task(taskData);

  ASSERT_FALSE(task.validation());
}