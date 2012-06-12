
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_PACKAGED_KERNEL_HPP
#define OCLM_PACKAGED_KERNEL_HPP

#include <oclm/config.hpp>

#include <string>

#include <oclm/range.hpp>
#include <oclm/program.hpp>
#include <oclm/buffer.hpp>

namespace oclm {

    template <typename Sig>
    struct packaged_kernel;

    template <typename T0, typename T1, typename T2>
    struct packaged_kernel<void(T0, T1, T2)>
    {
        typedef T0 t0_type;
        typedef T1 t1_type;
        typedef T2 t2_type;

        typedef typename result_of::make_buffer<T0>::type buffer0_type;
        typedef typename result_of::make_buffer<T1>::type buffer1_type;
        typedef typename result_of::make_buffer<T2>::type buffer2_type;

        program p_;
        std::string kernel_name_;
        ranges_type ranges_;
        buffer0_type t0_;
        buffer1_type t1_;
        buffer2_type t2_;

        packaged_kernel(program p, std::string kernel_name, ranges_type const & ranges, T0 & t0, T1 & t1, T2 & t2)
            : p_(p)
            , kernel_name_(kernel_name)
            , ranges_(ranges)
            , t0_(make_buffer(t0))
            , t1_(make_buffer(t1))
            , t2_(make_buffer(t2))
        {}

    };

}

#endif
