// Copyright (c) 2013 Erik Schnetter
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "matrix_hpx.hh"

#include "algorithms.hh"



HPX_REGISTER_COMPONENT_MODULE();



HPX_REGISTER_MINIMAL_COMPONENT_FACTORY
(hpx::components::simple_component<vector_t_server>, vector_t_factory);

void vector_t_server::axpy(double alpha, const hpx::id_type& x)
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  ::axpy(alpha, *fx.get(), *data);
}
void vector_t_server::copy(const hpx::id_type& x)
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  ::copy(*fx.get(), *data);
}
void vector_t_server::gemv(bool trans, double alpha, const hpx::id_type& a,
                           const hpx::id_type& x,
                           double beta)
{
  ::scal(beta, *data);
  auto ytmp =
    hpx::async(matrix_t_server::gemv_process_action(), a, trans, alpha, x);
  auto fytmp = hpx::async(vector_t_server::get_data_action(), ytmp.get());
  ::axpy(1.0, *fytmp.get(), *data);
  // TODO: delete ytmp
}
double vector_t_server::nrm2_process() const
{
  return ::nrm2_process(*data);
}
void vector_t_server::scal(double alpha)
{
  ::scal(alpha, *data);
}

hpx::future<void>
vector_t_client::gemv(bool trans, double alpha, const matrix_t_client& a,
                      const vector_t_client& x,
                      double beta)
  const
{
  return hpx::async(vector_t_server::gemv_action(), get_gid(),
                    trans, alpha, a.get_gid(), x.get_gid(), beta);
}



HPX_REGISTER_MINIMAL_COMPONENT_FACTORY
(hpx::components::simple_component<matrix_t_server>, matrix_t_factory);

void matrix_t_server::axpy(bool trans, double alpha, const hpx::id_type& a)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  ::axpy(trans, alpha, *fa.get(), *data);
}
void matrix_t_server::copy(bool transa, const hpx::id_type& a)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  ::copy(transa, *fa.get(), *data);
}
void matrix_t_server::gemm(bool transa, bool transb, double alpha,
                           const hpx::id_type& a, const hpx::id_type& b,
                           double beta)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  auto fb = hpx::async(matrix_t_server::get_data_action(), b);
  ::gemm(transa, transb, alpha, *fa.get(), *fb.get(), beta, *data);
}
hpx::id_type matrix_t_server::gemv_process(bool trans, double alpha,
                                           const hpx::id_type& x)
  const
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  vector_t_client y = vector_t_client::create(hpx::find_here(), data->NI);
  ::gemv(trans, alpha, *data, *fx.get(), 0.0, *y.get_ptr());
  return y.get_gid();
}
double matrix_t_server::nrm2_process() const
{
  return ::nrm2_process(*data);
}
void matrix_t_server::scal(double alpha)
{
  ::scal(alpha, *data);
}
