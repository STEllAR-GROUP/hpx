// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "tests.hh"

#include "algorithms.hh"
#include "block_matrix.hh"
#include "matrix.hh"

#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>



void test_dense()
{
  hpx::id_type here = hpx::find_here();
  std::cout << "test_dense: running on " << here << std::endl;
  
  const double alpha = 2.0, beta = 3.0;
  std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl;
  
  const ptrdiff_t NI=4, NJ=3, NK=2;
  std::cout << "NI=" << NI << ", NJ=" << NJ << ", NK=" << NK << std::endl;
  
  vector_t x(NJ);
  for (ptrdiff_t j=0; j<NJ; ++j) x(j) = j + 1;
  std::cout << "x=" << x << std::endl;
  vector_t y(NI);
  for (ptrdiff_t i=0; i<NI; ++i) y(i) = i + 2;
  std::cout << "y=" << y << std::endl;
  vector_t z(NI);
  for (ptrdiff_t i=0; i<NI; ++i) z(i) = i + 3;
  std::cout << "z=" << z << std::endl;
  
  matrix_t a(NI,NJ);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    for (ptrdiff_t j=0; j<NJ; ++j)
      a(i,j) = n++ + 4;
  std::cout << "a=" << a << std::endl;
  matrix_t b(NI,NK);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    for (ptrdiff_t k=0; k<NK; ++k)
      b(i,k) = n++ + 5;
  std::cout << "b=" << b << std::endl;
  matrix_t c(NK,NJ);
  for (ptrdiff_t k=0, n=0; k<NK; ++k)
    for (ptrdiff_t j=0; j<NJ; ++j)
      c(k,j) = n++ + 6;
  std::cout << "c=" << c << std::endl;
  
  std::cout << std::endl;
  
  
  
  vector_t yy(NI), zz(NI);
  matrix_t aa(NI,NJ);
  
  copy(z, zz);
  axpy(alpha, y, zz);
  std::cout << "axpy: alpha y + z = " << zz << std::endl;
  axpy(-1.0, vector_t({7,10,13,16}), zz);
  std::cout << "   (error = " << nrm2(zz) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(zz), 0.0);
  
  copy(y, yy);
  gemv(false, alpha, a, x, beta, yy);
  std::cout << "gemv: alpha a x + beta y = " << yy << std::endl;
  axpy(-1.0, vector_t({70,109,148,187}), yy);
  std::cout << "   (error = " << nrm2(yy) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(yy), 0.0);
  
  copy(false, a, aa);
  gemm(false, false, alpha, b, c, beta, aa);
  std::cout << "gemm: alpha b c + beta a = " << aa << std::endl;
  axpy(false,
       -1.0,
       matrix_t({{180,205,230},{249,282,315},{318,359,400},{387,436,485}}),
       aa);
  std::cout << "   (error = " << nrm2(aa) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(aa), 0.0);
  
  std::cout << std::endl;
}



void test_blocked()
{
  int nlocs = hpx::get_num_localities().get();
  std::vector<hpx::id_type> locs = hpx::find_all_localities();
  hpx::id_type here = hpx::find_here();
  std::cout << "test_blocked: running on " << here << std::endl;
  
  const double alpha = 2.0, beta = 3.0;
  std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl;
  
  const ptrdiff_t NI=10, NJ=6, NK=6;
  std::cout << "NI=" << NI << ", NJ=" << NJ << ", NK=" << NK << std::endl;
  
  const ptrdiff_t BI = 3;
  const ptrdiff_t istr0[BI] = {1, 4, 9};
  const ptrdiff_t istr1[BI] = {2, 6, 10};
  hpx::id_type ilocs[BI];
  for (std::ptrdiff_t i=0; i<BI; ++i) ilocs[i] = locs[i % nlocs];
  auto istr = std::make_shared<structure_t>(NI, BI, istr0, istr1, ilocs);
  std::cout << "istr=" << *istr << std::endl;
  
  const ptrdiff_t BJ = 2;
  const ptrdiff_t jstr0[BJ] = {0, 4};
  const ptrdiff_t jstr1[BJ] = {2, 5};
  hpx::id_type jlocs[BJ];
  for (std::ptrdiff_t j=0; j<BJ; ++j) jlocs[j] = locs[(j+1) % nlocs];
  auto jstr = std::make_shared<structure_t>(NJ, BJ, jstr0, jstr1, jlocs);
  std::cout << "jstr=" << *jstr << std::endl;
  
  const ptrdiff_t BK = 1;
  const ptrdiff_t kstr0[BK] = {1};
  const ptrdiff_t kstr1[BK] = {3};
  hpx::id_type klocs[BK];
  for (std::ptrdiff_t k=0; k<BK; ++k) klocs[k] = locs[(k+2) % nlocs];
  auto kstr = std::make_shared<structure_t>(NK, BK, kstr0, kstr1, klocs);
  std::cout << "kstr=" << *kstr << std::endl;
  
  block_vector_t x(jstr);
  for (ptrdiff_t j=0, n=0; j<NJ; ++j) {
    if (x.str->find(j) >= 0) {
      x.set_elt(j, n++ + 1);
    }
  }
  std::cout << "x=" << x << std::endl;
  
  block_vector_t y(istr);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    if (y.str->find(i) >= 0)
      y.set_elt(i, n++ + 2);
  std::cout << "y=" << y << std::endl;
  
  block_vector_t z(istr);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    if (z.str->find(i) >= 0)
      z.set_elt(i, n++ + 3);
  std::cout << "z=" << z << std::endl;
  
  block_matrix_t a(istr,jstr);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    if (a.istr->find(i) >= 0)
      for (ptrdiff_t j=0; j<NJ; ++j)
        if (a.jstr->find(j) >= 0)
          a.set_elt(i,j, n++ + 4);
  std::cout << "a=" << a << std::endl;
  
  block_matrix_t b(istr,kstr);
  for (ptrdiff_t i=0, n=0; i<NI; ++i)
    if (b.istr->find(i) >= 0)
      for (ptrdiff_t k=0; k<NK; ++k)
        if (b.jstr->find(k) >= 0)
          b.set_elt(i,k, n++ + 5);
  std::cout << "b=" << b << std::endl;
  
  block_matrix_t c(kstr,jstr);
  for (ptrdiff_t k=0, n=0; k<NK; ++k)
    if (c.istr->find(k) >= 0)
      for (ptrdiff_t j=0; j<NJ; ++j)
        if (c.jstr->find(j) >= 0)
          c.set_elt(k,j, n++ + 6);
  std::cout << "c=" << c << std::endl;
  
  std::cout << std::endl;
  
  
  
  block_vector_t yy(istr), zz(istr);
  block_matrix_t aa(istr,jstr);
  
  copy(z, zz);
  axpy(alpha, y, zz);
  std::cout << "axpy: alpha y + z = " << zz << std::endl;
  axpy(-1.0, block_vector_t(zz.str, {{1,{7}}, {4,{10,13}}, {9,{16}}}), zz);
  std::cout << "   (error = " << nrm2(zz) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(zz), 0.0);
  
  copy(y, yy);
  gemv(false, alpha, a, x, beta, yy);
  std::cout << "gemv: alpha a x + beta y = " << yy << std::endl;
  axpy(-1.0, block_vector_t(yy.str, {{1,{70}}, {4,{109,148}}, {9,{187}}}), yy);
  std::cout << "   (error = " << nrm2(yy) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(yy), 0.0);
  
  copy(false, a, aa);
  gemm(false, false, alpha, b, c, beta, aa);
  std::cout << "gemm: alpha b c + beta a = " << aa << std::endl;
  axpy(false, -1.0,
       block_matrix_t(aa.istr, aa.jstr,
                      {{
                          {{1,0}, {{180,205}}},
                          {{1,4}, {{230}}}
                        },{
                          {{4,0}, {{249,282},{318,359}}},
                          {{4,4}, {{315},{400}}}
                        },{
                          {{9,0}, {{387,436}}},
                          {{9,4}, {{485}}}
                        }}),
       aa);
  std::cout << "   (error = " << nrm2(aa) << ")" << std::endl;
  HPX_TEST_EQ(nrm2(aa), 0.0);
  
  std::cout << std::endl;
}



int report_errors()
{
  return hpx::util::report_errors();
}
