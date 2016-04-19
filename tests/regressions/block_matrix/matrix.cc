// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "matrix.hh"



std::ostream& operator<<(std::ostream& os, const vector_t& x)
{
  os << "[";
  for (std::ptrdiff_t i=0; i<x.N; ++i) {
    if (i != 0) os << ",";
    os << x(i);
  }
  os << "]";
  return os;
}



matrix_t::matrix_t(std::initializer_list<std::initializer_list<double>> a):
  NI(a.size()), NJ(a.end()==a.begin() ? 0 : a.begin()->size()), elts(NI*NJ)
{
  assert(std::ptrdiff_t(a.size()) == NI);
  std::ptrdiff_t i = 0;
  for (auto row: a) {
    assert(std::ptrdiff_t(row.size()) == NJ);
    std::ptrdiff_t j = 0;
    for (auto elt: row) {
      (*this)(i,j) = elt;
      ++j;
    }
    ++i;
  }
}

std::ostream& operator<<(std::ostream& os, const matrix_t& a)
{
  os << "[";
  for (std::ptrdiff_t i=0; i<a.NI; ++i) {
    if (i != 0) os << ",";
    os << "[";
    for (std::ptrdiff_t j=0; j<a.NJ; ++j) {
      if (j != 0) os << ",";
      os << a(i,j);
    }
    os << "]";
  }
  os << "]";
  return os;
}
