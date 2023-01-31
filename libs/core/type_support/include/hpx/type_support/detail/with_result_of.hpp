//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <utility>

namespace hpx::util::detail {

    // Based on
    // https://quuxplusone.github.io/blog/2018/05/17/super-elider-round-2/ and
    // https://akrzemi1.wordpress.com/2018/05/16/rvalues-redefined/.
    //
    // Useful for emplacing non-copyable and -movable types into e.g. variants.
    // The wrapper delays the construction of the non-copyable and -movable type
    // until it is required for conversion, and guaranteed copy elision in C++17
    // ensures that a copy constructor is not called when returning from the
    // conversion operator.
    template <typename F>
    class with_result_of_t
    {
        F&& f;

    public:
        using type = decltype(std::declval<F&&>()());

        explicit with_result_of_t(F&& f) noexcept
          : f(HPX_FORWARD(F, f))
        {
        }

        operator type() noexcept
        {
            return HPX_FORWARD(F, f)();
        }
    };

    template <typename F>
    with_result_of_t<F> with_result_of(F&& f) noexcept
    {
        return with_result_of_t<F>(HPX_FORWARD(F, f));
    }
}    // namespace hpx::util::detail
