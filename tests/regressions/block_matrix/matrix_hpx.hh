// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MATRIX_HPX_HH
#define MATRIX_HPX_HH

#include "matrix.hh"

#include <hpx/hpx.hpp>
#include <hpx/include/components.hpp>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>



struct matrix_t_client;

struct vector_t_server:
  public hpx::components::simple_component_base<vector_t_server>
{
  std::shared_ptr<vector_t> data;
  
  vector_t_server(std::ptrdiff_t N): data(std::make_shared<vector_t>(N)) {}
  vector_t_server& operator=(const vector_t_server&) = delete;
  // Temporarily, to allow creating a remote object from local data
  vector_t_server(const vector_t& x): data(std::make_shared<vector_t>(x)) {}
  // We don't really want these
  vector_t_server() { assert(0); }
  
  std::shared_ptr<vector_t> get_data() { return data; }
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, get_data);
  
  double get_elt(std::ptrdiff_t i) const { return (*data)(i); }
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, get_elt);
  void set_elt(std::ptrdiff_t i, double x) { (*data)(i) = x; }
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, set_elt);
  
  void axpy(double alpha, const hpx::id_type& x);
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, axpy);
  void copy(const hpx::id_type& x);
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, copy);
  void gemv(bool trans, double alpha, const hpx::id_type& a,
            const hpx::id_type& x,
            double beta);
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, gemv);
  double nrm2_process() const;
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, nrm2_process);
  void scal(double alpha);
  HPX_DEFINE_COMPONENT_ACTION(vector_t_server, scal);
};

struct vector_t_client:
  hpx::components::client_base<vector_t_client, vector_t_server>
{
  typedef hpx::components::client_base<vector_t_client, vector_t_server>
    base_type;

  vector_t_client() {}
  vector_t_client(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}

  std::shared_ptr<vector_t> get_ptr() const
  {
    return hpx::get_ptr<vector_t_server>(get_gid()).get()->data;
  }
  hpx::future<std::shared_ptr<vector_t>> get_data() const
  {
    return hpx::async(vector_t_server::get_data_action(), get_gid());
  }
  
  double get_elt(std::ptrdiff_t i) const
  {
    return vector_t_server::get_elt_action()(get_gid(), i);
  }
  void set_elt(std::ptrdiff_t i, double x) const
  {
    return vector_t_server::set_elt_action()(get_gid(), i, x);
  }

  hpx::future<void> axpy(double alpha, const vector_t_client& x) const
  {
    return hpx::async(vector_t_server::axpy_action(), get_gid(),
                      alpha, x.get_gid());
  }
  hpx::future<void> copy(const vector_t_client& x) const
  {
    return hpx::async(vector_t_server::copy_action(), get_gid(), x.get_gid());
  }
  hpx::future<void> gemv(bool trans, double alpha, const matrix_t_client& a,
                         const vector_t_client& x,
                         double beta) const;
  hpx::future<double> nrm2_process() const
  {
    return hpx::async(vector_t_server::nrm2_process_action(), get_gid());
  }
  hpx::future<void> scal(double alpha) const
  {
    return hpx::async(vector_t_server::scal_action(), get_gid(), alpha);
  }
};



struct matrix_t_server:
  public hpx::components::simple_component_base<matrix_t_server>
{
  std::shared_ptr<matrix_t> data;
  
  matrix_t_server(std::ptrdiff_t NI, std::ptrdiff_t NJ):
    data(std::make_shared<matrix_t>(NI,NJ))
  {
  }
  matrix_t_server& operator=(const matrix_t_server&) = delete;
  // Temporarily, to allow creating a remote object from local data
  matrix_t_server(const matrix_t& a): data(std::make_shared<matrix_t>(a)) {}
  // We don't really want these
  matrix_t_server() { assert(0); }
  
  std::shared_ptr<matrix_t> get_data() { return data; }
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, get_data);
  
  double get_elt(std::ptrdiff_t i, std::ptrdiff_t j) const
  {
    return (*data)(i,j);
  }
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, get_elt);
  void set_elt(std::ptrdiff_t i, std::ptrdiff_t j, double x)
  {
    (*data)(i,j) = x;
  }
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, set_elt);
  
  void axpy(bool trans, double alpha, const hpx::id_type& a);
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, axpy);
  void copy(bool transa, const hpx::id_type& a);
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, copy);
  void gemm(bool transa, bool transb, double alpha, const hpx::id_type& a,
            const hpx::id_type& b, double beta);
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, gemm);
  hpx::id_type gemv_process(bool trans, double alpha, const hpx::id_type& x)
    const;
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, gemv_process);
  double nrm2_process() const;
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, nrm2_process);
  void scal(double alpha);
  HPX_DEFINE_COMPONENT_ACTION(matrix_t_server, scal);
};

struct matrix_t_client:
  hpx::components::client_base<matrix_t_client, matrix_t_server>
{
  typedef hpx::components::client_base<matrix_t_client, matrix_t_server>
    base_type;

  matrix_t_client() {}
  matrix_t_client(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}

  std::shared_ptr<matrix_t> get_ptr() const
  {
    return hpx::get_ptr<matrix_t_server>(get_gid()).get()->data;
  }
  hpx::future<std::shared_ptr<matrix_t>> get_data() const
  {
    return hpx::async(matrix_t_server::get_data_action(), get_gid());
  }
  
  double get_elt(std::ptrdiff_t i, std::ptrdiff_t j) const
  {
    return matrix_t_server::get_elt_action()(get_gid(), i, j);
  }
  void set_elt(std::ptrdiff_t i, std::ptrdiff_t j, double x) const
  {
    return matrix_t_server::set_elt_action()(get_gid(), i, j, x);
  }
  
  hpx::future<void> axpy(bool trans, double alpha, const matrix_t_client& a)
    const
  {
    return hpx::async(matrix_t_server::axpy_action(), get_gid(),
                      trans, alpha, a.get_gid());
  }
  hpx::future<void> copy(bool transa, const matrix_t_client& a) const
  {
    return hpx::async(matrix_t_server::copy_action(), get_gid(),
                      transa, a.get_gid());
  }
  hpx::future<void> gemm(bool transa, bool transb, double alpha,
                         const matrix_t_client& a,
                         const matrix_t_client& b,
                         double beta) const
  {
    return hpx::async(matrix_t_server::gemm_action(), get_gid(),
                      transa, transb, alpha, a.get_gid(), b.get_gid(), beta);
  }
  // hpx::future<vector_t_client> gemv_process(bool trans, double alpha,
  //                                           const vector_t_client& x) const
  // {
  //   return hpx::async(matrix_t_server::gemv_process_action(), get_gid(),
  //                     trans, alpha, x.get_gid());
  // }
  hpx::future<double> nrm2_process() const
  {
    return hpx::async(matrix_t_server::nrm2_process_action(), get_gid());
  }
  hpx::future<void> scal(double alpha) const
  {
    return hpx::async(matrix_t_server::scal_action(), get_gid(), alpha);
  }
};

#endif // #ifndef MATRIX_HPX_HH
