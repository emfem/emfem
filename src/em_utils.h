#ifndef _EM_UTILS_H_
#define _EM_UTILS_H_ 1

#include <petsc.h>

#include <string>

std::string string_format(const char*, ...);
std::string parse_string(const std::string &s);

class LogEventHelper {
public:
  LogEventHelper(PetscLogEvent ple) : ple_(ple) { PetscLogEventBegin(ple_, 0, 0, 0, 0); }
  ~LogEventHelper() { PetscLogEventEnd(ple_, 0, 0, 0, 0); }

private:
  PetscLogEvent ple_;
};

class LogStageHelper {
public:
  LogStageHelper(const std::string &name) {
    PetscLogStage pls;
    PetscLogStageRegister(name.c_str(), &pls);
    PetscLogStagePush(pls);
  }
  ~LogStageHelper() { PetscLogStagePop(); }
};

#endif
