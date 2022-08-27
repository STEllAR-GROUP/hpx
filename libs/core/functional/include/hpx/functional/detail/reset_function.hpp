//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/function.hpp>
#include <hpx/functional/move_only_function.hpp>

namespace hpx { namespace util { namespace detail {

    template <typename Sig, bool Serializable>
    inline void reset_function(hpx::function<Sig, Serializable>& f)
    {
        f.reset();
    }

    template <typename Sig, bool Serializable>
    inline void reset_function(hpx::move_only_function<Sig, Serializable>& f)
    {
        f.reset();
    }

    template <typename Function>
    inline void reset_function(Function& f)
    {
        f = Function();
    }
}}}    // namespace hpx::util::detail
