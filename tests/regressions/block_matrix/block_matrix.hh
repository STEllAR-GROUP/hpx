// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BLOCK_MATRIX_HH
#define BLOCK_MATRIX_HH

#include "defs.hh"
#include "matrix.hh"
#include "matrix_hpx.hh"

#include <hpx/hpx.hpp>

#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>



struct structure_t {
  const std::ptrdiff_t N, B;
  const std::vector<std::ptrdiff_t> begin, end;
  const std::vector<hpx::id_type> locs;
  bool invariant() const;
  structure_t(std::ptrdiff_t N,
              std::ptrdiff_t B,
              const std::ptrdiff_t* begin, const std::ptrdiff_t* end,
              const hpx::id_type* locs):
    N(N), B(B), begin(begin, begin+B), end(end, end+B), locs(locs, locs+B)
  {
    assert(invariant());
  }
  operator std::string() const { return mkstr(*this); }
  bool operator==(const structure_t& str) const { return this == &str; }
  std::ptrdiff_t size(std::ptrdiff_t b) const
  {
    assert(b>=0 && b<B);
    return end[b] - begin[b];
  }
  std::ptrdiff_t find(std::ptrdiff_t i) const;
};

std::ostream& operator<<(std::ostream& os, const structure_t& str);



struct block_vector_t {
  template<typename T> using IL = std::initializer_list<T>;
  template<typename S, typename T> using P = std::pair<S,T>;
  boost::shared_ptr<structure_t> str;
  std::vector<vector_t_client> elts;
  block_vector_t(boost::shared_ptr<structure_t> str);
  block_vector_t(boost::shared_ptr<structure_t> str, IL<P<int, IL<double>>> x);
  operator std::string() const { return mkstr(*this); }
  const vector_t_client& block(std::ptrdiff_t b) const
  {
    assert(b>=0 && b<str->B);
    return elts[b];
  }
  double operator()(std::ptrdiff_t i) const
  {
    assert(i>=0 && i<str->N);
    auto b = str->find(i);
    static const double zero = 0.0;
    if (b < 0) return zero;
    return block(b).get_elt(i - str->begin[b]);
  }
  void set_elt(std::ptrdiff_t i, double x)
  {
    assert(i>=0 && i<str->N);
    auto b = str->find(i);
    assert(b >= 0);
    return block(b).set_elt(i - str->begin[b], x);
  }
};

std::ostream& operator<<(std::ostream& os, const block_vector_t& x);



struct block_matrix_t {
  template<typename T> using IL = std::initializer_list<T>;
  template<typename S, typename T> using P = std::pair<S,T>;
  boost::shared_ptr<structure_t> istr, jstr; // interpretation: row, column
  std::vector<matrix_t_client> elts;
  block_matrix_t(boost::shared_ptr<structure_t> istr,
                 boost::shared_ptr<structure_t> jstr);
  block_matrix_t(boost::shared_ptr<structure_t> istr,
                 boost::shared_ptr<structure_t> jstr,
                 IL<IL<P<P<int,int>, IL<IL<double>>>>> a);
  operator std::string() const { return mkstr(*this); }
  const matrix_t_client& block(std::ptrdiff_t ib, std::ptrdiff_t jb) const
  {
    assert(ib>=0 && ib<istr->B && jb>=0 && jb<=jstr->B);
    return elts[ib+istr->B*jb];
  }
  double operator()(std::ptrdiff_t i, std::ptrdiff_t j) const
  {
    assert(i>=0 && i<istr->N && j>=0 && j<jstr->N);
    auto ib = istr->find(i);
    auto jb = jstr->find(j);
    static const double zero = 0.0;
    if (ib < 0 || jb < 0) return zero;
    return block(ib,jb).get_elt(i - istr->begin[ib], j - jstr->begin[jb]);
  }
  void set_elt(std::ptrdiff_t i, std::ptrdiff_t j, double x)
  {
    assert(i>=0 && i<istr->N && j>=0 && j<jstr->N);
    auto ib = istr->find(i);
    auto jb = jstr->find(j);
    assert(ib >= 0 && jb >= 0);
    block(ib,jb).set_elt(i - istr->begin[ib], j - jstr->begin[jb], x);
  }
};

std::ostream& operator<<(std::ostream& os, const block_matrix_t& a);

#endif // #ifndef BLOCK_MATRIX_HH
