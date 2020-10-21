// Copyright (c) 2013 Erik Schnetter
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

#include "matrix_hpx.hpp"

#include "algorithms.hpp"



HPX_REGISTER_COMPONENT_MODULE();



HPX_REGISTER_MINIMAL_COMPONENT_FACTORY
(hpx::components::simple_component<vector_t_server>, vector_t_factory);

void vector_t_server::axpy(double alpha, const hpx::id_type& x)
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  auto fx_result = fx.get();
  ::axpy(alpha, *fx_result, *data);
}
void vector_t_server::copy(const hpx::id_type& x)
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  auto fx_result = fx.get();
  ::copy(*fx_result, *data);
}
void vector_t_server::gemv(bool trans, double alpha, const hpx::id_type& a,
                           const hpx::id_type& x,
                           double beta)
{
  ::scal(beta, *data);
  auto ytmp =
    hpx::async(matrix_t_server::gemv_process_action(), a, trans, alpha, x);
  hpx::id_type ytmp_id = ytmp.get();
  hpx::shared_future<std::shared_ptr<vector_t>> fytmp =
    hpx::async(vector_t_server::get_data_action(), ytmp_id);
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
  return hpx::async(vector_t_server::gemv_action(), get_id(),
                    trans, alpha, a.get_id(), x.get_id(), beta);
}



HPX_REGISTER_MINIMAL_COMPONENT_FACTORY
(hpx::components::simple_component<matrix_t_server>, matrix_t_factory);

void matrix_t_server::axpy(bool trans, double alpha, const hpx::id_type& a)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  auto fa_result = fa.get();
  ::axpy(trans, alpha, *fa_result, *data);
}
void matrix_t_server::copy(bool transa, const hpx::id_type& a)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  auto fa_result = fa.get();
  ::copy(transa, *fa_result, *data);
}
void matrix_t_server::gemm(bool transa, bool transb, double alpha,
                           const hpx::id_type& a, const hpx::id_type& b,
                           double beta)
{
  auto fa = hpx::async(matrix_t_server::get_data_action(), a);
  auto fb = hpx::async(matrix_t_server::get_data_action(), b);
  auto fa_result = fa.get();
  auto fb_result = fb.get();
  ::gemm(transa, transb, alpha, *fa_result, *fb_result, beta, *data);
}
hpx::id_type matrix_t_server::gemv_process(bool trans, double alpha,
                                           const hpx::id_type& x)
  const
{
  auto fx = hpx::async(vector_t_server::get_data_action(), x);
  auto fx_result = fx.get();
  vector_t_client y = hpx::new_<vector_t_client>(hpx::find_here(), data->NI);
  auto y_ptr = y.get_ptr();
  ::gemv(trans, alpha, *data, *fx_result, 0.0, *y_ptr);
  return y.get_id();
}
double matrix_t_server::nrm2_process() const
{
  return ::nrm2_process(*data);
}
void matrix_t_server::scal(double alpha)
{
  ::scal(alpha, *data);
}
#endif
