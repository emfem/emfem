#ifndef _EM_DEFS_H_
#define _EM_DEFS_H_ 1

#include <petsc.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <complex>
#include <tuple>

typedef std::complex<double> Complex;

typedef std::tuple<int, int> TupleII;

const double PI = 3.14159265358979323846;
const double MU = 4 * PI * 1E-7;
const Complex II = Complex(0.0, 1.0);

const double RTOD = 180.0 / PI;
const double DTOR = PI / 180.0;

const double EPS = 1.0E-6;

const int TX_SIZE = 7;

typedef Eigen::Vector3d Point;
typedef Eigen::Vector3d VectorD;
typedef Eigen::Vector3cd VectorZ;

typedef Eigen::Matrix3d Tensor;

struct PETScBlockVector {
  Vec re, im;
};

enum AnisotropicForm { Isotropic = 0, Vertical = 1, Triaxial = 2, Arbitrary = 3 };

enum PolarType { XY_POLAR = -1, YX_POLAR = -2};

enum RefineStrategy { FixedNumber = 0, FixedFraction = 1 };

enum InnerPCType { Mixed = 0, AMS = 1, Direct = 2 };

enum DirectSolverType { MUMPS = 0, SUPERLUDIST = 1 };

enum ObsType {
  F_EX_RI = 111,
  F_EY_RI = 121,
  F_EZ_RI = 131,
  F_HX_RI = 141,
  F_HY_RI = 151,
  F_HZ_RI = 161,
  F_EX_AP = 112,
  F_EY_AP = 122,
  F_EZ_AP = 132,
  F_HX_AP = 142,
  F_HY_AP = 152,
  F_HZ_AP = 162,
  R_XY_AP = 212,
  R_YX_AP = 222,
  Z_XX_RI = 311,
  Z_XY_RI = 321,
  Z_YX_RI = 331,
  Z_YY_RI = 341,
  T_ZX_RI = 351,
  T_ZY_RI = 361,
  Z_XX_AP = 312,
  Z_XY_AP = 322,
  Z_YX_AP = 332,
  Z_YY_AP = 342
};

#define EM_ERR_USER -1

#endif
