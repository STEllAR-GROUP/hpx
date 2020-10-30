// Copyright (c) 2013 Erik Schnetter
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "defs.hpp"

#include <hpx/assert.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/shared_ptr.hpp>
#include <hpx/serialization/vector.hpp>

#include <cassert>
#include <cstddef>
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
  template <class Archive>
  void serialize(Archive& ar, unsigned int)
  {
      // clang-format off
    ar & N;
    ar & elts;
      // clang-format on
  }

  explicit vector_t(std::ptrdiff_t N): N(N), elts(N) {}
  explicit vector_t(std::initializer_list<double> x): N(x.size()), elts(x) {}
  // We don't really want these
  vector_t() = default;
  vector_t(const vector_t&) = default;
  vector_t& operator=(const vector_t&) { HPX_ASSERT(0); return *this; }

  operator std::string() const { return mkstr(*this); }
  const double& operator()(std::ptrdiff_t i) const
  {
    HPX_ASSERT(i>=0 && i<N);
    return elts[i];
  }
  double& operator()(std::ptrdiff_t i)
  {
    HPX_ASSERT(i>=0 && i<N);
    return elts[i];
  }
};

std::ostream& operator<<(std::ostream& os, const vector_t& x);



struct matrix_t {
  std::ptrdiff_t NI, NJ;        // interpretation: row, column
  std::vector<double> elts;

  friend class hpx::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, unsigned int)
  {
      // clang-format off
    ar & NI & NJ;
    ar & elts;
      // clang-format on
  }

  matrix_t(std::ptrdiff_t NI, std::ptrdiff_t NJ): NI(NI), NJ(NJ), elts(NI*NJ) {}
  explicit matrix_t(std::initializer_list<std::initializer_list<double>> a);
  // We don't really want these
  matrix_t() = default;
  matrix_t(const matrix_t&) = default;

  operator std::string() const { return mkstr(*this); }
  const double& operator()(std::ptrdiff_t i, std::ptrdiff_t j) const
  {
    HPX_ASSERT(i>=0 && i<NI && j>=0 && j<NJ);
    return elts[i+NI*j];
  }
  double& operator()(std::ptrdiff_t i, std::ptrdiff_t j)
  {
    HPX_ASSERT(i>=0 && i<NI && j>=0 && j<NJ);
    return elts[i+NI*j];
  }
};

std::ostream& operator<<(std::ostream& os, const matrix_t& a);

#endif
