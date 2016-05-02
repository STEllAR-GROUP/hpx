// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "block_matrix.hh"

#include "matrix.hh"

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>

#include <cassert>
#include <cstdlib>



bool structure_t::invariant() const
{
  if (N < 0) return false;
  if (B < 0) return false;
  if (B > 0) {
    if (begin[0] < 0) return false;
    for (std::ptrdiff_t b=0; b<B-1; ++b) {
      if (end[b] < begin[b]) return false;
      if (end[b] > begin[b+1]) return false;
    }
    if (end[B-1] > N) return false;
  }
  return true;
}

std::ptrdiff_t structure_t::find(std::ptrdiff_t i) const
{
  assert(i>=0 && i<N);
  if (B == 0) return -1;
  std::ptrdiff_t b0 = 0, b1 = B-1;
  auto loopinv = [&]() { return b0>=0 && b1<B && b0<=b1; };
  auto loopvar = [&]() { return b1 - b0; };
  assert(loopinv());
  (void)loopinv;
  std::ptrdiff_t old_loopvar = loopvar();
  while (b0 < b1 && i>=begin[b0] && i<end[b1]) {
    std::ptrdiff_t b = (b0 + b1)/2;
    if (i < end[b]) {
      b1 = b;
    } else {
      b0 = b + 1;
    }
    assert(loopinv());
    auto next_loopvar = loopvar();
    assert(next_loopvar >= 0 && next_loopvar < old_loopvar);
      old_loopvar = next_loopvar;
  }
  if (b0 == b1 && i>=begin[b0] && i<end[b1]) {
    // found
    return b0;
  }
  // not found
  return -1;
}

std::ostream& operator<<(std::ostream& os, const structure_t& str)
{
  os << "{";
  for (std::ptrdiff_t b=0; b<str.B; ++b) {
    if (b != 0) os << ",";
    os << str.begin[b] << ":" << str.end[b];
  }
  os << "}";
  return os;
}



block_vector_t::block_vector_t(std::shared_ptr<structure_t> str):
  str(str), elts(str->B)
{
  for (std::ptrdiff_t b=0; b<str->B; ++b) {
    elts[b] = vector_t_client::create(str->locs[b], str->size(b));
  }
}

block_vector_t::block_vector_t(std::shared_ptr<structure_t> str,
                               IL<P<int, IL<double>>> x):
  block_vector_t(str)
{
  assert(std::ptrdiff_t(x.size()) == str->B);
  std::ptrdiff_t b = 0;
  for (auto blk: x) {
    assert(blk.first == str->begin[b]);
    assert(std::ptrdiff_t(blk.second.size()) == str->size(b));
    std::ptrdiff_t i = str->begin[b];
    for (auto elt: blk.second) {
      assert(str->find(i) >= 0);
      set_elt(i, elt);
      ++i;
    }
    ++b;
  }
}

std::ostream& operator<<(std::ostream& os, const block_vector_t& x)
{
  os << "{";
  for (std::ptrdiff_t b=0; b<x.str->B; ++b) {
    if (b != 0) os << ",";
    os << x.str->begin[b] << ":" << *x.block(b).get_data().get();
  }
  os << "}";
  return os;
}



block_matrix_t::block_matrix_t(std::shared_ptr<structure_t> istr,
                               std::shared_ptr<structure_t> jstr):
  istr(istr), jstr(jstr), elts(istr->B*jstr->B)
{
  for (std::ptrdiff_t jb=0; jb<jstr->B; ++jb) {
    for (std::ptrdiff_t ib=0; ib<istr->B; ++ib) {
      elts[ib+istr->B*jb] = matrix_t_client::create(istr->locs[ib],
                                 istr->size(ib), jstr->size(jb));
    }
  }
}

block_matrix_t::block_matrix_t(std::shared_ptr<structure_t> istr,
                               std::shared_ptr<structure_t> jstr,
                               IL<IL<P<P<int,int>, IL<IL<double>>>>> a):
  block_matrix_t(istr, jstr)
{
  assert(std::ptrdiff_t(a.size()) == istr->B);
  std::ptrdiff_t ib = 0;
  for (auto irow: a) {
    assert(std::ptrdiff_t(irow.size()) == jstr->B);
    std::ptrdiff_t jb = 0;
    for (auto blk: irow) {
      assert(blk.first.first == istr->begin[ib]);
      assert(blk.first.second == jstr->begin[jb]);
      std::ptrdiff_t i = istr->begin[ib];
      for (auto row: blk.second) {
        std::ptrdiff_t j = jstr->begin[jb];
        for (auto elt: row) {
          assert(istr->find(i) >= 0 && jstr->find(j) >= 0);
          set_elt(i,j, elt);
          ++j;
        }
        ++i;
      }
      ++jb;
    }
    ++ib;
  }
}

std::ostream& operator<<(std::ostream& os, const block_matrix_t& a)
{
  os << "{";
  for (std::ptrdiff_t ib=0; ib<a.istr->B; ++ib) {
    if (ib != 0) os << ",";
    os << "{";
    for (std::ptrdiff_t jb=0; jb<a.jstr->B; ++jb) {
      if (jb != 0) os << ",";
      os << "(" << a.istr->begin[ib] << "," << a.jstr->begin[jb] << "):"
         << *a.block(ib,jb).get_data().get();
    }
    os << "}";
  }
  os << "}";
  return os;
}
