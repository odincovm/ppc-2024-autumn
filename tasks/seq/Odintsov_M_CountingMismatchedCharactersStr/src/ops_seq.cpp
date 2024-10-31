
#include "seq/Odintsov_M_CountingMismatchedCharactersStr/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential::validation() {
  internal_order_test();
  // Проверка на то, что у нас 2 строки на входе и одно число на выходе
  bool ans_out = (taskData->inputs_count[0] == 2);
  bool ans_in = (taskData->outputs_count[0] == 1);
  return (ans_in) && (ans_out);
}
bool Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential::pre_processing() {
  internal_order_test();
  // инициализация инпута
  if (strlen(reinterpret_cast<char*>(taskData->inputs[0])) >= strlen(reinterpret_cast<char*>(taskData->inputs[1]))) {
    input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
    input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
  } else {
    input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
    input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
  }
  // Инициализация ответа
  ans = 0;
  return true;
}
bool Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential::run() {
  internal_order_test();
  for (int i = 0; i < strlen(input[0]); i++) {
    if (i < strlen(input[1])) {
      if (input[0][i] != input[1][i]) {
        ans += 2;
      }
    } else {
      ans += 1;
    }
  }
  return true;
}
bool Odintsov_M_CountingMismatchedCharactersStr_seq::CountingCharacterSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = ans;
  return true;
}
