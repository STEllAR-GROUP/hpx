//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_MAKE_CONTINUATION_HPP
#define HPX_ACTIONS_MAKE_CONTINUATION_HPP

#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/actions/set_lco_value_continuation.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/actions/continuation2_impl.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx {
    inline hpx::actions::set_lco_value_continuation
    make_continuation()
    {
        return hpx::actions::set_lco_value_continuation();
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename util::decay<Cont>::type
    >
    make_continuation(Cont && cont)
    {
        typedef typename util::decay<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            std::forward<Cont>(cont), hpx::find_here());
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename util::decay<Cont>::type
    >
    make_continuation(Cont && f, hpx::naming::id_type const& target)
    {
        typedef typename util::decay<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            std::forward<Cont>(f), target);
    }

    template <typename Cont, typename F>
    inline typename std::enable_if<
        !std::is_same<
            typename util::decay<F>::type,
            hpx::naming::id_type
        >::value,
        hpx::actions::continuation2_impl<
            typename util::decay<Cont>::type,
            typename util::decay<F>::type
        >
    >::type
    make_continuation(Cont && cont, F && f)
    {
        typedef typename util::decay<Cont>::type cont_type;
        typedef typename util::decay<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            std::forward<Cont>(cont), hpx::find_here(), std::forward<F>(f));
    }

    template <typename Cont, typename F>
    inline hpx::actions::continuation2_impl<
        typename util::decay<Cont>::type,
        typename util::decay<F>::type
    >
    make_continuation(Cont && cont, hpx::naming::id_type const& target,
        F && f)
    {
        typedef typename util::decay<Cont>::type cont_type;
        typedef typename util::decay<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            std::forward<Cont>(cont), target, std::forward<F>(f));
    }
}

#endif
