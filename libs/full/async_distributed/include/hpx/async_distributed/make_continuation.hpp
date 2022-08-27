//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_distributed/continuation2_impl.hpp>
#include <hpx/async_distributed/continuation_impl.hpp>
#include <hpx/async_distributed/set_lco_value_continuation.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <type_traits>
#include <utility>

namespace hpx {

    inline hpx::actions::set_lco_value_continuation make_continuation()
    {
        return {};
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<typename std::decay<Cont>::type>
    make_continuation(Cont&& cont)
    {
        using cont_type = typename std::decay<Cont>::type;
        return hpx::actions::continuation_impl<cont_type>(
            HPX_FORWARD(Cont, cont),
            naming::get_id_from_locality_id(agas::get_locality_id()));
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<typename std::decay<Cont>::type>
    make_continuation(Cont&& f, hpx::id_type const& target)
    {
        using cont_type = typename std::decay<Cont>::type;
        return hpx::actions::continuation_impl<cont_type>(
            HPX_FORWARD(Cont, f), target);
    }

    template <typename Cont, typename F>
    inline typename std::enable_if<
        !std::is_same<typename std::decay<F>::type, hpx::id_type>::value,
        hpx::actions::continuation2_impl<typename std::decay<Cont>::type,
            typename std::decay<F>::type>>::type
    make_continuation(Cont&& cont, F&& f)
    {
        using cont_type = typename std::decay<Cont>::type;
        using function_type = typename std::decay<F>::type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            HPX_FORWARD(Cont, cont),
            naming::get_id_from_locality_id(agas::get_locality_id()),
            HPX_FORWARD(F, f));
    }

    template <typename Cont, typename F>
    inline hpx::actions::continuation2_impl<typename std::decay<Cont>::type,
        typename std::decay<F>::type>
    make_continuation(Cont&& cont, hpx::id_type const& target, F&& f)
    {
        using cont_type = typename std::decay<Cont>::type;
        using function_type = typename std::decay<F>::type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            HPX_FORWARD(Cont, cont), target, HPX_FORWARD(F, f));
    }
}    // namespace hpx
