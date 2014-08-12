//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_PACKAGED_ACTION_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM)
#define HPX_LCOS_PACKAGED_ACTION_CONSTRUCTORS_DIRECT_JUL_01_2008_0116PM

#include <hpx/util/move.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/packaged_action_constructors_direct.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/packaged_action_constructors_direct_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/packaged_action_constructors_direct.hpp"))                      \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply(BOOST_SCOPED_ENUM(launch) /*policy*/, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            // local, direct execution
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));

            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg))))
            );
        }
        else {
            // remote execution
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply(BOOST_SCOPED_ENUM(launch) /*policy*/, naming::address&& addr,
        naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

        if (addr.locality_ == hpx::get_locality()) {
            // local, direct execution
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));

            (*this->impl_)->set_data(
                std::move(action_type::execute_function(addr.address_,
                    util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg))))
            );
        }
        else {
            // remote execution
            hpx::applier::detail::apply_c_cb<action_type>(
                std::move(addr), this->get_gid(), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    packaged_action(naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : apply_logger_("packaged_action_direct::apply")
    {
        LLCO_(info) << "packaged_action::packaged_action("
                    << hpx::actions::detail::get_action_name<action_type>()
                    << ", "
                    << gid
                    << ") args(" << (N + 1) << ")";
        apply(launch::all, gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

#undef N

#endif
