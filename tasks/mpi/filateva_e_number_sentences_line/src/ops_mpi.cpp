// Filateva Elizaveta Number_of_sentences_per_line
#include "mpi/filateva_e_number_sentences_line/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

int filateva_e_number_sentences_line_mpi::countSentences(std::string line) {
  int count = 0;
  for (long unsigned int i = 0; i < line.size(); ++i) {
    if (line[i] == '.' || line[i] == '?' || line[i] == '!') {
      ++count;
    }
  }
  return count;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  line = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
  sentence_count = 0;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::run() {
  internal_order_test();
  sentence_count = countSentences(line);
  if (!line.empty() && line.back() != '.' && line.back() != '?' && line.back() != '!') {
    ++sentence_count;
  }
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    line = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
  }

  sentence_count = 0;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remains = 0;
  int local_sentence_count;
  if (world.rank() == 0 && world.size() > 1) {
    delta = line.size() / (world.size() - 1);
    remains = line.size() % (world.size() - 1);
  } else if (world.rank() == 0 && world.size() == 1) {
    remains = line.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 0; proc < (world.size() - 1); proc++) {
      world.send(proc + 1, 0, line.data() + proc * delta + remains, delta);
    }
    local_line = std::string(line.begin(), line.begin() + remains);
  } else {
    local_line = std::string(delta, '*');
    world.recv(0, 0, local_line.data(), delta);
  }

  local_sentence_count = countSentences(local_line);
  if (world.rank() == 0 && !line.empty() && line.back() != '.' && line.back() != '?' && line.back() != '!') {
    ++local_sentence_count;
  }
  reduce(world, local_sentence_count, sentence_count, std::plus(), 0);
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
  }
  return true;
}
