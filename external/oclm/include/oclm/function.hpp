
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_FUNCTION_HPP
#define OCLM_FUNCTION_HPP

#include <string>

#include <oclm/packaged_kernel.hpp>
#include <oclm/program.hpp>
#include <oclm/range.hpp>

namespace oclm {
    struct function
    {
        template <typename T0>
        function(program const & p, std::string const & kernel_name, T0 const & t0)
            : p_(p)
            , kernel_name_(kernel_name)
        {
            ranges_.set(t0);
        }

        template <typename T0, typename T1>
        function(program const & p, std::string const & kernel_name, T0 const & t0, T1 const & t1)
            : p_(p)
            , kernel_name_(kernel_name)
        {
            ranges_.set(t0);
            ranges_.set(t1);
        }

        template <typename T0, typename T1, typename T2>
        function(program const & p, std::string const & kernel_name, T0 const & t0, T1 const & t1, T2 const & t2)
            : p_(p)
            , kernel_name_(kernel_name)
        {
            ranges_.set(t0);
            ranges_.set(t1);
            ranges_.set(t2);
        }

        template <typename T0, typename T1, typename T2>
        packaged_kernel<void(T0, T1, T2)> operator()(T0 & t0, T1 & t1, T2 & t2) const
        {
            return packaged_kernel<void(T0, T1, T2)>(p_, kernel_name_, ranges_, t0, t1, t2);
        }
        
        program p_;
        std::string kernel_name_;
        ranges_type ranges_;
    };
}

#endif
