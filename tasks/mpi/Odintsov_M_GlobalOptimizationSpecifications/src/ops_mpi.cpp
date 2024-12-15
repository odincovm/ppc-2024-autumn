
#include "mpi/Odintsov_M_GlobalOptimizationSpecifications/include/ops_mpi.hpp"

using namespace std::chrono_literals;

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::
    satisfies_constraints(double x, double y, int number_constraint) {
  double check = constraint[number_constraint * 3] * x + constraint[number_constraint * 3 + 1] * y -
                 constraint[number_constraint * 3 + 2];
  return check <= 0;
}

double
Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::calculate_function(
    double x, double y) {
  return (x - funct[0]) * (x - funct[0]) + (y - funct[1])*(y-funct[1]);
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::validation() {
  internal_order_test();
  // Проверка на то что количество ограничений не 0
  if (taskData->outputs_count[0] == 0) return false;
  // Проверка на корректость введения версии;
  if ((taskData->inputs_count[1] != 1) && (taskData->inputs_count[1] != 0)) return false;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::pre_processing() {
  internal_order_test();
  // инициализация дополнительных данных
  step = *reinterpret_cast<double*>(taskData->inputs[3]);

  for (int i = 0; i < 4; i++) {
    area.push_back(reinterpret_cast<double*>(taskData->inputs[0])[i]);
  }

  count_constraint = taskData->inputs_count[0];
  ver = taskData->inputs_count[1];

  // инициализация функций
  for (int i = 0; i < 2; i++) {
    funct.push_back(reinterpret_cast<double*>(taskData->inputs[1])[i]);
  }

  for (int i = 0; i < count_constraint * 3; i++) {
    constraint.push_back(reinterpret_cast<double*>(taskData->inputs[2])[i]);
  }

  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::run() {
  internal_order_test();
  if (ver == 0)
    ans = 999999999999999;
  else
    ans = -999999999999999;
  double mX = 0;
  double mY = 0;
  double minX = -100;
  double minY = -100;
  double current_step = step;                                // Текущий шаг сетки
  double tolerance = 1e-6;                                   // Точность выхода
  double previous_ans = std::numeric_limits<double>::max();  // Предыдущее значение минимальной функции
  // Главный цикл уточнения сетки
  while (current_step >= tolerance) {
    double local_minX = minX;
    double local_minY = minY;

    // Перебираем сетку с текущим шагом
    for (double x = area[0]; x < area[1]; x += current_step) {
      for (double y = area[2]; y < area[3]; y += current_step) {
        // Проверяем на ограничения
        bool is_point_correct = true;
        for (int i = 0; i < count_constraint; i++) {
          is_point_correct = satisfies_constraints(x, y, i);
          if (!is_point_correct) break;
        }
        // Если точка удовлетворяет ограничениям
        if (is_point_correct) {
          double value = calculate_function(x, y);
          if (ver == 0) {  // Минимизация
            if (value < ans) {
              ans = value;
              mX = x;
              mY = y;
              local_minX = x;
              local_minY = y;
            }
          } else if (ver == 1) {  // Максимизация
            ans = std::max(ans, value);
          }
        }
      }
    }

    // Если точность достигнута, завершаем
    if (std::abs(previous_ans - ans) < tolerance) {
      break;
    }
    // Уточняем границы области вокруг локального минимума
    area[0] = std::max(local_minX - 2 * current_step, area[0]);
    area[1] = std::min(local_minX + 2 * current_step, area[1]);
    area[2] = std::max(local_minY - 2 * current_step, area[2]);
    area[3] = std::min(local_minY + 2 * current_step, area[3]);
    // Уменьшаем шаг сетки
    current_step /= 2.0;
    previous_ans = ans;
  }

  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = ans;
  return true;
}
// Паралельная версия
bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::
    satisfies_constraints(double x, double y, int number_constraint) {
  double check = local_constraint[number_constraint * 3] * x + local_constraint[number_constraint * 3 + 1] * y -
                 local_constraint[number_constraint * 3 + 2];
  return check <= 0;
}

double Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::calculate_function(
    double x, double y) {
  return (x - funct[0]) * (x - funct[0]) + (y - funct[1]) * (y - funct[1]);
}
bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::validation() {
  internal_order_test();
  if (com.rank() == 0) {
    // Проверка на то что количество ограничений не 0
    if (taskData->outputs_count[0] == 0) return false;
    // Проверка на корректость введения версии;
    if ((taskData->inputs_count[1] != 1) && (taskData->inputs_count[1] != 0)) return false;
  }
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    step = *reinterpret_cast<double*>(taskData->inputs[3]);
    area.resize(4);
    for (int i = 0; i < 4; i++) {
      area[i] = (reinterpret_cast<double*>(taskData->inputs[0])[i]);
    }

    count_constraint = taskData->inputs_count[0];
    ver = taskData->inputs_count[1];
    // инициализация функций
    for (int i = 0; i < 2; i++) {
      funct.push_back(reinterpret_cast<double*>(taskData->inputs[1])[i]);
    }
    for (int i = 0; i < count_constraint * 3; i++) {
      constraint.push_back(reinterpret_cast<double*>(taskData->inputs[2])[i]);
    }
  }
  loc_constr_size = 0;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::run() {
  internal_order_test();

  // Сброс переменных
  ans = (ver == 0) ? 999999999999999 : -999999999999999;

  // Определение и Передача количества ограничений на поток
  if (com.rank() == 0) {
    loc_constr_size = std::max(1, (count_constraint + com.size() - 1) / com.size());
  }

  broadcast(com, count_constraint, 0);
  broadcast(com, loc_constr_size, 0);
  broadcast(com, step, 0);

  fflush(stdout);

  if (com.rank() != 0) {
    area.resize(4);
  }

  broadcast(com, area.data(), area.size(), 0);

  // printf("Rank[%d]: After broadcasting area\n", com.rank());
  // fflush(stdout);
  //  Передача ограничений потокам (без случая когда потоков больше)
  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      if (pr * loc_constr_size < count_constraint) {
        std::vector<double> send;
        for (int i = 0; i < 3 * loc_constr_size; i++) {
          send.push_back(constraint[pr * loc_constr_size * 3 + i]);
        }
        com.send(pr, 0, send.data(), send.size());
      }
    }
    // Заполнение 0 потока
    for (int i = 0; i < 3 * loc_constr_size; i++) {
      local_constraint.push_back(constraint[i]);
    }
  } else if (com.rank() < count_constraint) {
    // printf("soft lock 1\n");
    // fflush(stdout);
    std::vector<double> buffer(loc_constr_size * 3, 0);

    com.recv(0, 0, buffer.data(), buffer.size());
    // printf("soft lock 2\n");
    // fflush(stdout);
    local_constraint.insert(local_constraint.end(), buffer.begin(), buffer.end());
  }

  // printf("Rang %i Area %f %f %f %f \n", com.rank(), area[0], area[1], area[2], area[3]);
  // fflush(stdout);
  //  Вычисления
  double current_step = step;
  double tolerance = 1e-6;
  double previous_ans = std::numeric_limits<double>::max();
  std::vector<double> loc_area;
  for (int i = 0; i < 4; i++) {
    loc_area.push_back(area[i]);
  }
  if (!local_constraint.empty()) {
    while (current_step >= tolerance) {
      double local_minX = loc_area[0];
      double local_minY = loc_area[2];
      // fflush(stdout);
      //  Локальные вычисления на каждом потоке
      for (double x = loc_area[0]; x < loc_area[1]; x += current_step) {
        for (double y = loc_area[2]; y < loc_area[3]; y += current_step) {
          // Локальная проверка ограничений

          int loc_flag = 1;
          for (int i = 0; i < loc_constr_size; i++) {
            if (!satisfies_constraints(x, y, i)) {
              loc_flag = 0;
              break;
            }
          }

          // Объединяем данные ограничений в 0 поток
          gather(com, loc_flag, is_corret, 0);

          // Проверка и вычисление функции (выполняется только на потоке 0)
          if (com.rank() == 0) {
            bool flag = true;
            int sz = is_corret.size();
            for (int i = 0; i < sz; i++) {
              if (is_corret[i] == 0) {
                flag = false;
                break;
              }
            }
            if (flag) {
              double value = calculate_function(x, y);
              if (ver == 0) {  // Минимизация
                if (value < ans) {
                  ans = value;
                  local_minX = x;
                  local_minY = y;
                }
              } else if (ver == 1) {  // Максимизация
                ans = std::max(ans, value);
              }
            }
          }
        }
      }

      // Поток 0 обновляет границы области и проверяет сходимость
      if (com.rank() == 0) {
        // Проверяем на сходимость

        if ((std::abs(previous_ans - ans) < tolerance)) {
          current_step = -1;  // Завершаем цикл
        }
        std::vector<double> new_area = loc_area;

        // Уточняем границы
        new_area[0] = std::max(local_minX - 2 * current_step, area[0]);
        new_area[1] = std::min(local_minX + 2 * current_step, area[1]);
        new_area[2] = std::max(local_minY - 2 * current_step, area[2]);
        new_area[3] = std::min(local_minY + 2 * current_step, area[3]);

        loc_area = new_area;
        previous_ans = ans;
      }

      // Передаем новые границы области и шаг всем потокам
      broadcast(com, loc_area.data(), area.size(), 0);
      broadcast(com, current_step, 0);
    }
  }

  return true;
}
bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) reinterpret_cast<double*>(taskData->outputs[0])[0] = ans;
  return true;
}