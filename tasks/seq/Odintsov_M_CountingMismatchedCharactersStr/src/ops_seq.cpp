
#include "seq/Odintsov_M_CountingMismatchedCharactersStr/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

using namespace Odintsov_M_CountingMismatchedCharactersStr_seq;

  bool CountingCharacterSequential::validation() {
    internal_order_test();
    // ѕроверка на то, что у нас 2 строки на входе и одно число на выходе
    bool ans_out = (taskData->inputs_count[0] == 2);
    bool ans_in = (taskData->outputs_count[0] == 1);
    return (ans_in) && (ans_out);
  }
  bool CountingCharacterSequential::pre_processing() {
    internal_order_test();
    //инициализаци€ инпута
    input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
    input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
    
    // »нициализаци€ ответа
    ans = 0;
    return true;
  }
  bool CountingCharacterSequential::run() {
      // спросить €вл€ютс€ считать ли повтор€ющиес€ символы как один или как несколько
    internal_order_test();
    char tmp;
    int count=0;
    bool first = true;
    for (int i = 0; i < strlen(input[0]); i++) {
      first = true;
      tmp = input[0][i];
      for (int j = 0; j < strlen(input[1]); j++) {
        if (tmp == input[1][j]) {
          count += 1;
          if (first) {
            count += 1;
            first = false;
          }
        }
      }
    }
    ans = (strlen(input[0]) + strlen(input[1])) - count;
    std::this_thread::sleep_for(20ms);
    return true;
  }
  bool CountingCharacterSequential::post_processing() {
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = ans;
    return true;
  }

  



