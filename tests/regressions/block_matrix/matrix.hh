// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MATRIX_HH
#define MATRIX_HH

#include "defs.hh"

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>



struct vector_t {
  std::ptrdiff_t N;
  std::vector<double> elts;
  
  friend class hpx::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, unsigned int version)
  {
    ar & N;
    ar & elts;
  }
  
  vector_t(std::ptrdiff_t N): N(N), elts(N) {}
  vector_t(std::initializer_list<double> x): N(x.size()), elts(x) {}
  // We don't really want these
  vector_t() = default;
  vector_t(const vector_t&) = default;
  vector_t& operator=(const vector_t&) { assert(0); return *this; }
  
  operator std::string() const { return mkstr(*this); }
  const double& operator()(std::ptrdiff_t i) const
  {
    assert(i>=0 && i<N);
    return elts[i];
  }
  double& operator()(std::ptrdiff_t i)
  {
    assert(i>=0 && i<N);
    return elts[i];
  }
};

std::ostream& operator<<(std::ostream& os, const vector_t& x);



struct matrix_t {
  std::ptrdiff_t NI, NJ;        // interpretation: row, column
  std::vector<double> elts;
  
  friend class hpx::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, unsigned int version)
  {
    ar & NI & NJ;
    ar & elts;
  }
  
  matrix_t(std::ptrdiff_t NI, std::ptrdiff_t NJ): NI(NI), NJ(NJ), elts(NI*NJ) {}
  matrix_t(std::initializer_list<std::initializer_list<double>> a);
  // We don't really want these
  matrix_t() = default;
  matrix_t(const matrix_t&) = default;
  
  operator std::string() const { return mkstr(*this); }
  const double& operator()(std::ptrdiff_t i, std::ptrdiff_t j) const
  {
    assert(i>=0 && i<NI && j>=0 && j<NJ);
    return elts[i+NI*j];
  }
  double& operator()(std::ptrdiff_t i, std::ptrdiff_t j)
  {
    assert(i>=0 && i<NI && j>=0 && j<NJ);
    return elts[i+NI*j];
  }
};

std::ostream& operator<<(std::ostream& os, const matrix_t& a);

#endif // #ifndef MATRIX_HH
