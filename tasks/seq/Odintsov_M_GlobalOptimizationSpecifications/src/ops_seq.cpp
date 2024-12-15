
#include "seq/Odintsov_M_GlobalOptimizationSpecifications/include/ops_seq.hpp"

bool Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::satisfies_constraints(
    double x, double y, int number_constraint) {
  double check = constraint[number_constraint * 3] * x + constraint[number_constraint * 3 + 1] * y -
                 constraint[number_constraint * 3 + 2];
  return check <= 0;
}

double Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::calculate_function(
    double x, double y) {
  return (x - funct[0]) * (x - funct[0]) + (y - funct[1]) * (y - funct[1]);
}

bool Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::validation() {
  internal_order_test();
  // Проверка на то что количество ограничений не 0
  if (taskData->outputs_count[0] == 0) return false;
  // Проверка на корректость введения версии;
  if ((taskData->inputs_count[1] != 1) && (taskData->inputs_count[1] != 0)) return false;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::pre_processing() {
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

bool Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::run() {
  internal_order_test();
  if (ver == 0)
    ans = 999999999999999;
  else
    ans = -999999999999999;
  // double minX = -100;
  // double minY = -100;
  double current_step = step;                                // Текущий шаг сетки
  double tolerance = 1e-6;                                   // Точность выхода
  double previous_ans = std::numeric_limits<double>::max();  // Предыдущее значение минимальной функции
  int scale_factor = static_cast<int>(1.0 / current_step);
  // Главный цикл уточнения сетки
  while (current_step >= tolerance) {
    double local_minX = area[0];
    double local_minY = area[2];

    // Масштабируем границы для целочисленного цикла
    int int_minX = static_cast<int>(area[0] * scale_factor);
    int int_maxX = static_cast<int>(area[1] * scale_factor);
    int int_minY = static_cast<int>(area[2] * scale_factor);
    int int_maxY = static_cast<int>(area[3] * scale_factor);

    // Перебираем сетку с текущим шагом
    for (int x = int_minX; x < int_maxX; x++) {
      for (int y = int_minY; y < int_maxY; y++) {
        double real_x = x / static_cast<double>(scale_factor);
        double real_y = y / static_cast<double>(scale_factor);

        // Проверяем на ограничения
        bool is_point_correct = true;
        for (int i = 0; i < count_constraint; i++) {
          is_point_correct = satisfies_constraints(real_x, real_y, i);
          if (!is_point_correct) break;
        }
        // Если точка удовлетворяет ограничениям
        if (is_point_correct) {
          double value = calculate_function(real_x, real_y);
          if (ver == 0) {  // Минимизация
            if (value < ans) {
              ans = value;
              local_minX = real_x;
              local_minY = real_y;
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

    // Пересчитываем масштабированные границы
    int_minX = static_cast<int>(area[0] * scale_factor);
    int_maxX = static_cast<int>(area[1] * scale_factor);
    int_minY = static_cast<int>(area[2] * scale_factor);
    int_maxY = static_cast<int>(area[3] * scale_factor);

    // Уменьшаем шаг сетки
    current_step /= 2.0;
    previous_ans = ans;
  }

  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = ans;

  return true;
}
