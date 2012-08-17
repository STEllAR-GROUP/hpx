//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PREPROCESSED_LCOS_WAIT_N_HPP)
#define HPX_PREPROCESSED_LCOS_WAIT_N_HPP

#if HPX_WAIT_ARGUMENT_LIMIT  <= 5
#include <hpx/lcos/preprocessed/wait_n_5.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 10
#include <hpx/lcos/preprocessed/wait_n_10.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 15
#include <hpx/lcos/preprocessed/wait_n_15.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 20
#include <hpx/lcos/preprocessed/wait_n_20.hpp>
/*
#elif HPX_WAIT_ARGUMENT_LIMIT <= 25
#include <hpx/lcos/preprocessed/wait_n_25.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 30
#include <hpx/lcos/preprocessed/wait_n_30.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 35
#include <hpx/lcos/preprocessed/wait_n_35.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 40
#include <hpx/lcos/preprocessed/wait_n_40.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 45
#include <hpx/lcos/preprocessed/wait_n_45.hpp>
#elif HPX_WAIT_ARGUMENT_LIMIT <= 50
#include <hpx/lcos/preprocessed/wait_n_50.hpp>
*/
#else
#error "HPX_WAIT_ARGUMENT_LIMIT out of bounds for preprocessed headers"
#endif

#endif
