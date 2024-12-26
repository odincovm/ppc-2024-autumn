#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>

#include "mpi/beskhmelnova_k_dinning_philosophers/include/dinning_philosophers.hpp"

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_world_size_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (num_philosophers >= 2) {
    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(dining_task.validation());
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_0_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_1_philosopher) {
  boost::mpi::communicator world;

  int num_philosophers = 1;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_negative_size_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = -1;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_2_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 2;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_3_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 3;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_4_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 4;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_with_8_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 8;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_think_function) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    dining_task.think();
    ASSERT_EQ(dining_task.state, beskhmelnova_k_dining_philosophers::THINKING);
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_eat_function) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    dining_task.eat();
    ASSERT_EQ(dining_task.state, beskhmelnova_k_dining_philosophers::EATING);
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_request_forks_function) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    dining_task.request_forks();
    ASSERT_EQ(dining_task.state, beskhmelnova_k_dining_philosophers::HUNGRY);
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
  } else {
    GTEST_SKIP();
  }
}

TEST(beskhmelnova_k_dinning_philosophers_mpi, Test_release_forks_function) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (dining_task.validation()) {
    ASSERT_TRUE(dining_task.pre_processing());
    dining_task.eat();
    ASSERT_EQ(dining_task.state, beskhmelnova_k_dining_philosophers::EATING);
    dining_task.release_forks();
    ASSERT_EQ(dining_task.state, beskhmelnova_k_dining_philosophers::THINKING);
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
  } else {
    GTEST_SKIP();
  }
}
