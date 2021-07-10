#ifndef _EM_PW_H_
#define _EM_PW_H_ 1

#include "em_defs.h"

class PlaneWaveFunction {
public:
  void initialize(const Eigen::VectorXd &ztop, const Eigen::VectorXd &lsig,
                  double freq) {
    int i;
    Complex gmogp, rjexp, Atop;
    Eigen::VectorXcd kk, Rp, expmgh;

    nlay_ = (int)ztop.size();
    ztop_ = ztop;
    lsig_ = lsig;

    omega_ = 2 * PI * freq;

    kk.resize(nlay_ + 1);
    Rp.resize(nlay_ + 1);
    expmgh.resize(nlay_ + 1);

    for (i = 1; i <= nlay_; ++i) {
      kk[i] = std::sqrt(-II * omega_ * MU * lsig_[i - 1]);
    }
    kk[0] = std::sqrt(-II * omega_ * MU * lsig_[0]);

    expmgh.setZero();
    for (i = 1; i <= nlay_ - 1; ++i) {
      expmgh[i] = std::exp(-kk[i] * (ztop_[i] - ztop_[i - 1]));
    }

    Rp.setZero();
    for (i = nlay_ - 1; i >= 1; --i) {
      gmogp = (-II * omega_ * MU * (lsig_[i - 1] - lsig_[i])) / std::pow(kk[i] + kk[i + 1], 2);
      rjexp = Rp[i + 1] * expmgh[i + 1];
      Rp[i] = (gmogp + rjexp) / (1.0 + gmogp * rjexp) * expmgh[i];
    }

    a.resize(nlay_ + 1);
    b.resize(nlay_ + 1);
    c.resize(nlay_ + 1);
    d.resize(nlay_ + 1);
    a.setZero();
    b.setZero();
    c.setZero();
    d.setZero();
    a[0] = Complex(1.0);
    c[0] = Complex(1.0);

    for (i = 1; i <= nlay_; ++i) {
      Atop = a[i - 1] + b[i - 1] * expmgh[i - 1];
      b[i] = Atop / (1.0 + Rp[i] * expmgh[i]);
      a[i] = b[i] * Rp[i];

      Atop = c[i - 1] + d[i - 1] * expmgh[i - 1];
      d[i] = Atop / (1.0 - Rp[i] * expmgh[i]);
      c[i] = -d[i] * Rp[i];
    }
  }

  void calculate_field(double z, Complex eh[4]) const {
    int i, ilay;
    Complex kk, expp, expm, exppdz, expmdz;

    ilay = 0;
    for (i = nlay_ - 1; i >= 0; --i) {
      if (z >= ztop_[i]) {
        ilay = i;
        break;
      }
    }

    kk = std::sqrt(-II * omega_ * MU * lsig_[ilay]);

    expp = Complex(0.0);
    expm = Complex(0.0);
    exppdz = Complex(0.0);
    expmdz = Complex(0.0);

    expm = std::exp(-kk * (z - ztop_[ilay]));
    expmdz = -kk * expm;
    if (ilay != (nlay_ - 1)) {
      expp = std::exp(kk * (z - ztop_[ilay + 1]));
      exppdz = kk * expp;
    }

    eh[0] = a[ilay + 1] * expp + b[ilay + 1] * expm;
    eh[1] = (a[ilay + 1] * exppdz + b[ilay + 1] * expmdz) / (II * MU * omega_);
    eh[2] = (c[ilay + 1] * exppdz + d[ilay + 1] * expmdz) / lsig_[ilay];
    eh[3] = c[ilay + 1] * expp + d[ilay + 1] * expm;

    if (std::abs(eh[0]) < 1E-80) {
      eh[0] = 0.0;
    }
    if (std::abs(eh[1]) < 1E-80) {
      eh[1] = 0.0;
    }
    if (std::abs(eh[2]) < 1E-80) {
      eh[2] = 0.0;
    }
    if (std::abs(eh[3]) < 1E-80) {
      eh[3] = 0.0;
    }
  }

  void calculate_efield(const Point &p, int mode, VectorZ &ep) const {
    Complex eh[4];

    ep.setZero();

    calculate_field(p[2], eh);
    if (mode == XY_POLAR) {
      ep[0] = std::conj(eh[0]);
    } else if (mode == YX_POLAR) {
      ep[1] = std::conj(eh[2]);
    }
  }

  void calculate_hfield(const Point &p, int mode, VectorZ &hp) const {
    Complex eh[4];

    hp.setZero();

    calculate_field(p[2], eh);
    if (mode == XY_POLAR) {
      hp[1] = std::conj(eh[1]);
    } else if (mode == YX_POLAR) {
      hp[0] = std::conj(eh[3]);
    }
  }

private:
  int nlay_;
  double omega_;
  Eigen::VectorXcd a, b, c, d;
  Eigen::VectorXd ztop_, lsig_;
};

#endif
