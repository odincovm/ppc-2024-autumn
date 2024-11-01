
#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

#include <thread>
#include <cstring>
using namespace std::chrono_literals;
using namespace Odintsov_M_CountingMismatchedCharactersStr_mpi;
// Последовательная версия
bool CountingCharacterMPISequential::validation() {
  internal_order_test();
  // Проверка на то, что у нас 2 строки на входе и одно число на выходе
  bool ans_out = (taskData->inputs_count[0] == 2);
  bool ans_in = (taskData->outputs_count[0] == 1);
  return (ans_in) && (ans_out);
}
bool CountingCharacterMPISequential::pre_processing() {
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
bool CountingCharacterMPISequential::run() {
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
bool CountingCharacterMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = ans;
  return true;
}
// Параллельная версия
bool CountingCharacterMPIParallel::validation() {
  internal_order_test();
  // Проверка на то, что у нас 2 строки на входе и одно число на выходе
  if (com.rank() == 0) {
    bool ans_out = (taskData->inputs_count[0] == 2);
    bool ans_in = (taskData->outputs_count[0] == 1);
    return (ans_in) && (ans_out);
  }
  return true;
}
// Сделать для не кратного числа потоков, и для разных длин
bool CountingCharacterMPIParallel::pre_processing() {
  internal_order_test();
  // Получение количества потоков
  int cout_n;
  int loc_size = 0;
  cout_n = com.size();

  // Инициализация в 0 поток
  if (com.rank() == 0) {
    // Инициализация loc_size;
    if (strlen(reinterpret_cast<char*>(taskData->inputs[0])) >= strlen(reinterpret_cast<char*>(taskData->inputs[1]))) {
      loc_size = strlen(reinterpret_cast<char*>(taskData->inputs[0])) / com.size();
    } else {
      loc_size = strlen(reinterpret_cast<char*>(taskData->inputs[1])) / com.size();
    }
  }

  broadcast(com, loc_size, 0);
  if (com.rank() == 0) {
    // инициализация инпута
    if (strlen(reinterpret_cast<char*>(taskData->inputs[0])) >= strlen(reinterpret_cast<char*>(taskData->inputs[1]))) {
      input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
      input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
    } else {
      input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
      input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
    }
    for (int pr = 1; pr < com.size(); pr++) {
      com.send(pr, 0, input[0] + pr * loc_size, loc_size);
      com.send(pr, 0, input[1] + pr * loc_size, loc_size);
    }
  }
  char* str1 = new char[loc_size + 1];
  char* str2 = new char[loc_size + 1];

  if (com.rank() == 0) {
    memcpy(str1, input[0], loc_size);
    memcpy(str2, input[1], loc_size);
    local_input.push_back(str1);
    local_input.push_back(str2);
  } else {
    com.recv(0, 0, str1, loc_size);
    com.recv(0, 0, str2, loc_size);
    local_input.push_back(str1);
    local_input.push_back(str2);
  }
  ans = 0;
  return true;
}
bool CountingCharacterMPIParallel::run() {
  internal_order_test();
  int loc_res = 0;
  //
  for (int i = 0; i < strlen(input[0]); i++) {
    if (i < strlen(input[1])) {
      if (input[0][i] != input[1][i]) {
        loc_res += 2;
      }
    } else {
      loc_res += 1;
    }
  }
  ans = loc_res;
  return true;
}

bool CountingCharacterMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = ans;
  }
  return true;
}