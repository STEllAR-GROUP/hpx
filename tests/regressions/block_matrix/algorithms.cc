// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "algorithms.hh"

#include "block_matrix.hh"
#include "matrix.hh"
#include "matrix_hpx.hh"

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>



// Level 1

// axpy: y = alpha x + y

void axpy(double alpha, const vector_t& x, vector_t& y)
{
  assert(x.N == y.N);
  if (alpha == 0.0) return;
  if (alpha == 1.0) {
    for (std::ptrdiff_t i=0; i<y.N; ++i) {
      y(i) += x(i);
    }
  } else {
    for (std::ptrdiff_t i=0; i<y.N; ++i) {
      y(i) += alpha * x(i);
    }
  }
}

void axpy(double alpha, const block_vector_t& x, block_vector_t& y)
{
  assert(x.str == y.str);
  std::vector<hpx::future<void>> fs;
  for (std::ptrdiff_t ib=0; ib<y.str->B; ++ib) {
    fs.push_back(y.block(ib).axpy(alpha, x.block(ib)));
  }
  hpx::wait_all(fs);
}



// copy: y = x

void copy(const vector_t& x, vector_t& y)
{
  assert(x.N == y.N);
  for (std::ptrdiff_t i=0; i<y.N; ++i) {
    y(i) = x(i);
  }
}

void copy(const block_vector_t& x, block_vector_t& y)
{
  assert(x.str == y.str);
  std::vector<hpx::future<void>> fs;
  for (std::ptrdiff_t ib=0; ib<y.str->B; ++ib) {
    fs.push_back(y.block(ib).copy(x.block(ib)));
  }
  hpx::wait_all(fs);
}



// nrm2: sqrt(x^T x)

namespace {
  double nrm2_init()
  {
    return 0.0;
  }
  double nrm2_accumulate(double val, double xi)
  {
    return val + xi * xi;
  }
  double nrm2_finalize(double val)
  {
    return std::sqrt(val);
  }
}

double nrm2_process(const vector_t& x)
{
  double val = nrm2_init();
  for (std::ptrdiff_t i=0; i<x.N; ++i) {
    val = nrm2_accumulate(val, x(i));
  }
  return val;
}

double nrm2(const vector_t& x)
{
  double val = nrm2_process(x);
  return nrm2_finalize(val);
}

double nrm2(const block_vector_t& x)
{
  std::vector<hpx::future<double>> fs;
  for (std::ptrdiff_t ib=0; ib<x.str->B; ++ib) {
    fs.push_back(x.block(ib).nrm2_process());
  }
  double val = nrm2_init();
  for (auto& f: fs) {
    val = nrm2_accumulate(val, f.get());
  }
  return nrm2_finalize(val);
}



// scal: x = alpha x

void scal(double alpha, vector_t& x)
{
  if (alpha == 1.0) return;
  if (alpha == 0.0) {
    for (std::ptrdiff_t i=0; i<x.N; ++i) {
      x(i) = 0.0;
    }
  } else {
    for (std::ptrdiff_t i=0; i<x.N; ++i) {
      x(i) *= alpha;
    }
  }
}

void scal(double alpha, block_vector_t& x)
{
  if (alpha == 1.0) return;
  std::vector<hpx::future<void>> fs;
  for (std::ptrdiff_t ib=0; ib<x.str->B; ++ib) {
    fs.push_back(x.block(ib).scal(alpha));
  }
  hpx::wait_all(fs);
}



// Level 2

// axpy: b = alpha a + b

void axpy(bool trans, double alpha, const matrix_t& a, matrix_t& b)
{
  if (alpha == 0.0) return;
  if (!trans) {
    assert(b.NI == a.NI);
    assert(b.NJ == a.NJ);
    if (alpha == 1.0) {
      for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<b.NI; ++i) {
          b(i,j) += a(i,j);
        }
      }
    } else {
      for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<b.NI; ++i) {
          b(i,j) += alpha * a(i,j);
        }
      }
    }
  } else {
    assert(b.NI == a.NJ);
    assert(b.NJ == a.NI);
    if (alpha == 1.0) {
      for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<b.NI; ++i) {
          b(i,j) += a(j,i);
        }
      }
    } else {
      for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<b.NI; ++i) {
          b(i,j) += alpha * a(j,i);
        }
      }
    }
  }
}

void axpy(bool trans, double alpha, const block_matrix_t& a,
          block_matrix_t& b)
{
  if (alpha == 0.0) return;
  std::vector<hpx::future<void>> fs;
  if (!trans) {
    assert(b.istr == a.istr);
    assert(b.jstr == a.jstr);
    for (std::ptrdiff_t jb=0; jb<b.jstr->B; ++jb) {
      for (std::ptrdiff_t ib=0; ib<b.istr->B; ++ib) {
        fs.push_back(b.block(ib,jb).axpy(trans, alpha, a.block(ib,jb)));
      }
    }
  } else {
    assert(b.istr == a.jstr);
    assert(b.jstr == a.istr);
    for (std::ptrdiff_t jb=0; jb<b.jstr->B; ++jb) {
      for (std::ptrdiff_t ib=0; ib<b.istr->B; ++ib) {
        fs.push_back(b.block(ib,jb).axpy(trans, alpha, a.block(jb,ib)));
      }
    }
  }
  hpx::wait_all(fs);
}



// copy: y = x

void copy(bool transa, const matrix_t& a, matrix_t& b)
{
  if (!transa) {
    assert(b.NI == a.NI);
    assert(b.NJ == a.NJ);
    for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
      for (std::ptrdiff_t i=0; i<b.NI; ++i) {
        b(i,j) = a(i,j);
      }
    }
  } else {
    assert(b.NI == a.NJ);
    assert(b.NJ == a.NI);
    for (std::ptrdiff_t j=0; j<b.NJ; ++j) {
      for (std::ptrdiff_t i=0; i<b.NI; ++i) {
        b(i,j) = a(j,i);
      }
    }
  }
}

void copy(bool transa, const block_matrix_t& a, block_matrix_t& b)
{
  std::vector<hpx::future<void>> fs;
  if (!transa) {
    assert(b.istr == a.istr);
    assert(b.jstr == a.jstr);
    for (std::ptrdiff_t jb=0; jb<b.jstr->B; ++jb) {
      for (std::ptrdiff_t ib=0; ib<b.istr->B; ++ib) {
        fs.push_back(b.block(ib,jb).copy(transa, a.block(ib,jb)));
      }
    }
  } else {
    assert(b.istr == a.jstr);
    assert(b.jstr == a.istr);
    for (std::ptrdiff_t jb=0; jb<b.jstr->B; ++jb) {
      for (std::ptrdiff_t ib=0; ib<b.istr->B; ++ib) {
        fs.push_back(b.block(ib,jb).copy(transa, a.block(jb,ib)));
      }
    }
  }
  hpx::wait_all(fs);
}



// gemv: y = alpha T[a] x + beta y

void gemv(bool trans, double alpha, const matrix_t& a, const vector_t& x,
          double beta, vector_t& y)
{
  scal(beta, y);
  if (alpha == 0.0) return;
  if (!trans) {
    assert(a.NJ == x.N);
    assert(a.NI == y.N);
    for (std::ptrdiff_t j=0; j<x.N; ++j) {
      double tmp = alpha * x(j);
      for (std::ptrdiff_t i=0; i<y.N; ++i) {
        y(i) += a(i,j) * tmp;
      }
    }
  } else {
    assert(a.NI == x.N);
    assert(a.NJ == y.N);
    for (std::ptrdiff_t j=0; j<y.N; ++j) {
      double tmp = 0.0;
      for (std::ptrdiff_t i=0; i<x.N; ++i) {
        tmp += a(i,j) * x(i);
      }
      y(j) += alpha * tmp;
    }
  }
}

void gemv(bool trans, double alpha,
          const block_matrix_t& a, const block_vector_t& x,
          double beta, block_vector_t& y)
{
  scal(beta, y);
  if (alpha == 0.0) return;
  if (!trans) {
    assert(a.jstr == x.str);
    assert(a.istr == y.str);
    block_vector_t xtmp(x.str);
    copy(x, xtmp);
    scal(alpha, xtmp);
    for (std::ptrdiff_t jb=0; jb<x.str->B; ++jb) {
      std::vector<hpx::future<void>> fs;
      for (std::ptrdiff_t ib=0; ib<y.str->B; ++ib) {
        fs.push_back(y.block(ib).gemv(trans, 1.0, a.block(ib,jb),
                                      xtmp.block(jb), 1.0));
      }
      hpx::wait_all(fs);
    }
  } else {
    // TODO
    assert(0);
#if 0
    assert(a.jstr == x.str);
    assert(a.istr == y.str);
    for (std::ptrdiff_t jb=0; jb<y.str->B; ++jb) {
      vector_t tmp(y.str->size(jb));
      scal(0.0, tmp);
      for (std::ptrdiff_t ib=0; ib<x.str->B; ++ib) {
        gemv(trans, 1.0, a.block(ib,jb), x.block(ib), 1.0, tmp);
      }
      axpy(alpha, tmp, y.block(jb));
    }
#endif
  }
}



// nrm2: sqrt(trace a^T a)

double nrm2_process(const matrix_t& a)
{
  double val = nrm2_init();
  for (std::ptrdiff_t j=0; j<a.NJ; ++j) {
    for (std::ptrdiff_t i=0; i<a.NI; ++i) {
      val = nrm2_accumulate(val, a(i,j));
    }
  }
  return val;
}

double nrm2(const matrix_t& a)
{
  double val = nrm2_process(a);
  return nrm2_finalize(val);
}

double nrm2(const block_matrix_t& a)
{
  std::vector<hpx::future<double>> fs;
  for (std::ptrdiff_t jb=0; jb<a.jstr->B; ++jb) {
    for (std::ptrdiff_t ib=0; ib<a.istr->B; ++ib) {
      fs.push_back(a.block(ib,jb).nrm2_process());
    }
  }
  double val = nrm2_init();
  for (auto& f: fs) {
    val = nrm2_accumulate(val, f.get());
  }
  return nrm2_finalize(val);
}



// scal: a = alpha a

void scal(double alpha, matrix_t& a)
{
  if (alpha == 1.0) return;
  if (alpha == 0.0) {
    for (std::ptrdiff_t j=0; j<a.NJ; ++j) {
      for (std::ptrdiff_t i=0; i<a.NI; ++i) {
        a(i,j) = 0.0;
      }
    }
  } else {
    for (std::ptrdiff_t j=0; j<a.NJ; ++j) {
      for (std::ptrdiff_t i=0; i<a.NI; ++i) {
        a(i,j) *= alpha;
      }
    }
  }
}

void scal(double alpha, block_matrix_t& a)
{
  if (alpha == 1.0) return;
  std::vector<hpx::future<void>> fs;
  for (std::ptrdiff_t jb=0; jb<a.jstr->B; ++jb) {
    for (std::ptrdiff_t ib=0; ib<a.istr->B; ++ib) {
      fs.push_back(a.block(ib,jb).scal(alpha));
    }
  }
  hpx::wait_all(fs);
}



// Level 3

// gemm: c = alpha T[a] T[b] + beta c

void gemm(bool transa, bool transb,
          double alpha, const matrix_t& a, const matrix_t& b,
          double beta, matrix_t& c)
{
  if (alpha == 0.0) {
    scal(beta, c);
    return;
  }
  if (!transb) {
    if (!transa) {
      // c = alpha a b + beta c
      assert(b.NI == a.NJ);
      assert(c.NI == a.NI);
      assert(c.NJ == b.NJ);
      for (std::ptrdiff_t j=0; j<c.NJ; ++j) {
        if (beta == 0.0) {
          for (std::ptrdiff_t i=0; i<c.NI; ++i) {
            c(i,j) = 0.0;
          }
        } else {
          for (std::ptrdiff_t i=0; i<c.NI; ++i) {
            c(i,j) *= beta;
          }
        }
        for (std::ptrdiff_t k=0; k<b.NI; ++k) {
          if (b(k,j) != 0.0) {
            double tmp = alpha * b(k,j);
            for (std::ptrdiff_t i=0; i<c.NI; ++i) {
              c(i,j) += tmp * a(i,k);
            }
          }
        }
      }
    } else {
      // c = alpha a^T b + beta c
      assert(b.NI == a.NI);
      assert(c.NI == a.NJ);
      assert(c.NJ == b.NJ);
      for (std::ptrdiff_t j=0; j<c.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<c.NI; ++i) {
          double tmp = 0.0;
          for (std::ptrdiff_t k=0; k<b.NI; ++k) {
            tmp += a(k,i) * b(k,j);
          }
          if (beta == 0.0) {
            c(i,j) = alpha * tmp;
          } else {
            c(i,j) = alpha * tmp + beta * c(i,j);
          }
        }
      }
    }
  } else {
    if (!transa) {
      // c = alpha a b^T + beta c
      assert(b.NJ == a.NJ);
      assert(c.NI == a.NI);
      assert(c.NJ == b.NI);
      for (std::ptrdiff_t j=0; j<c.NJ; ++j) {
        if (beta == 0.0) {
          for (std::ptrdiff_t i=0; i<c.NI; ++i) {
            c(i,j) = 0.0;
          }
        } else if (beta != 1.0) {
          for (std::ptrdiff_t i=0; i<c.NI; ++i) {
            c(i,j) *= beta;
          }
        }
        for (std::ptrdiff_t k=0; k<b.NJ; ++k) {
          if (b(j,k) != 0.0) {
            double tmp = alpha * b(j,k);
            for (std::ptrdiff_t i=0; i<c.NI; ++i) {
              c(i,j) += tmp * a(i,k);
            }
          }
        }
      }
    } else {
      // c = alpha a^T b^T + beta c
      assert(b.NJ == a.NI);
      assert(c.NI == a.NJ);
      assert(c.NJ == b.NI);
      for (std::ptrdiff_t j=0; j<c.NJ; ++j) {
        for (std::ptrdiff_t i=0; i<c.NI; ++i) {
          double tmp = 0.0;
          for (std::ptrdiff_t k=0; k<b.NJ; ++k) {
            tmp += a(k,i) * b(j,k);
          }
          if (beta == 0.0) {
            c(i,j) = alpha * tmp;
          } else {
            c(i,j) = alpha * tmp + beta * c(i,j);
          }
        }
      }
    }
  }
}

void gemm(bool transa, bool transb,
          double alpha, const block_matrix_t& a, const block_matrix_t& b,
          double beta, block_matrix_t& c)
{
  scal(beta, c);
  if (alpha == 0.0) return;
  if (!transb) {
    if (!transa) {
      // c = alpha a b + beta c
      assert(b.istr == a.jstr);
      assert(c.istr == a.istr);
      assert(c.jstr == b.jstr);
#if 0
      for (std::ptrdiff_t jb=0; jb<c.jstr->B; ++jb) {
        for (std::ptrdiff_t ib=0; ib<c.istr->B; ++ib) {
          scal(beta, c.block(ib,jb));
        }
        for (std::ptrdiff_t kb=0; kb<b.istr->B; ++kb) {
          matrix_t tmp(b.block(kb,jb));
          scal(alpha, tmp);
          for (std::ptrdiff_t ib=0; ib<c.istr->B; ++ib) {
            gemm(transa, transb, 1.0, a.block(ib,kb), tmp, 1.0, c.block(ib,jb));
          }
        }
      }
#endif
      block_matrix_t btmp(b.istr,b.jstr);
      copy(false, b, btmp);
      scal(alpha, btmp);
      for (std::ptrdiff_t jb=0; jb<c.jstr->B; ++jb) {
        for (std::ptrdiff_t kb=0; kb<b.istr->B; ++kb) {
          std::vector<hpx::future<void>> fs;
          for (std::ptrdiff_t ib=0; ib<c.istr->B; ++ib) {
            fs.push_back(c.block(ib,jb).gemm(transa, transb,
                                             1.0,
                                             a.block(ib,kb), btmp.block(kb,jb),
                                             1.0));
          }
          hpx::wait_all(fs);
        }
      }
    } else {
      // TODO
      assert(0);
    }
  } else {
    // TODO
    assert(0);
  }
}
