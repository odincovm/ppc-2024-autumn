#pragma warning(disable : 4996)
#include <gtest/gtest.h>

#include <vector>

#include "seq/Odintsov_M_CountingMismatchedCharactersStr/include/ops_seq.hpp"

TEST(Sequential_count, ans_8) {
  // Create data
  char* str1 = new char[6];
  char* str2 = new char[6];
  strcpy(str1,"qwert");
  strcpy(str2,"qello");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData//
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(8, out[0]);
}

TEST(Sequential_count, ans_0) {
  // Create data
  char* str1 = new char[6];
  char* str2 = new char[6];
  strcpy(str1, "qwert");
  strcpy(str2, "qwert");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(0, out[0]);
}
TEST(Sequential_count, ans_10) {
  // Create data
  char* str1 = new char[6];
  char* str2 = new char[6];
  strcpy(str1, "qwert");
  strcpy(str2, "asdfg");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(10, out[0]);
}
TEST(Sequential_count, ans_11) {
  // Create data
  char* str1 = new char[7];
  char* str2 = new char[6];
  strcpy(str1, "qwerta");
  strcpy(str2, "asdfg");
  std::vector<char*> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(11, out[0]);
}