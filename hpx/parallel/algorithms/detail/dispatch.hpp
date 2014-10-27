//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM)
#define HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/serialization/serialization.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
#define HPX_PARALLEL_DISPATCH(policy, ...)                                    \
    switch(detail::which(policy))                                             \
    {                                                                         \
    case detail::execution_policy_enum::sequential:                           \
        return call(*policy.get<sequential_execution_policy>(), __VA_ARGS__,  \
            boost::mpl::true_());                                             \
                                                                              \
    case detail::execution_policy_enum::sequential_task:                      \
        return call(seq, __VA_ARGS__, boost::mpl::true_());                   \
                                                                              \
    case detail::execution_policy_enum::parallel:                             \
        return call(*policy.get<parallel_execution_policy>(), __VA_ARGS__,    \
            boost::mpl::false_());                                            \
                                                                              \
    case detail::execution_policy_enum::parallel_task:                        \
        {                                                                     \
            parallel_task_execution_policy const& t =                         \
                *policy.get<parallel_task_execution_policy>();                \
            return call(par(t.get_executor(), t.get_chunk_size()),            \
                __VA_ARGS__, boost::mpl::false_());                           \
        }                                                                     \
                                                                              \
    case detail::execution_policy_enum::parallel_vector:                      \
        return call(*policy.get<parallel_vector_execution_policy>(),          \
            __VA_ARGS__, boost::mpl::false_());                               \
                                                                              \
    default:                                                                  \
        HPX_THROW_EXCEPTION(hpx::bad_parameter,                               \
            std::string("hpx::parallel::") + name_,                           \
            "Not supported execution policy");                                \
    }                                                                         \
    /**/

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Tuple>
    struct async_fused_wrapper
    {
        typedef typename parallel::v1::detail::algorithm_result<
            sequential_task_execution_policy, typename Derived::result_type
        >::type result_type;

        async_fused_wrapper(Tuple && t)
          : t_(std::move(t))
        {}

        BOOST_FORCEINLINE result_type operator()() const
        {
            return hpx::util::invoke_fused_r<result_type>(
                Derived(), std::move(t_));
        }

        typename hpx::util::tuple_decay<Tuple>::type t_;
    };

    template <typename Derived, typename Tuple>
    BOOST_FORCEINLINE async_fused_wrapper<
        Derived
      , typename hpx::util::decay<Tuple>::type
    >
    get_async_fused_wrapper(Tuple && t)
    {
        typedef async_fused_wrapper<
            Derived
          , typename hpx::util::decay<Tuple>::type
        > invoker_type;

        return invoker_type(std::forward<Tuple>(t));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
        typedef Result result_type;

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, 5, "hpx/parallel/algorithms/detail/dispatch.hpp"))                \
    /**/

#include BOOST_PP_ITERATE()

        explicit algorithm(char const* const name) : name_(name) {}

        char const* const name_;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int)
        {
        }
    };
}}}}

#undef HPX_PARALLEL_DISPATCH

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    template <typename ExPolicy, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename parallel::v1::detail::algorithm_result<
        ExPolicy, result_type
    >::type
    call(ExPolicy const& policy, HPX_ENUM_FWD_ARGS(N, Arg, arg),
        boost::mpl::true_) const
    {
        try {
            return parallel::v1::detail::algorithm_result<
                    ExPolicy, result_type
                >::get(Derived::sequential(
                    policy, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
        }
        catch (...) {
            parallel::v1::detail::handle_exception<ExPolicy>::call();
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename parallel::v1::detail::algorithm_result<
        sequential_task_execution_policy, result_type
    >::type
    operator()(sequential_task_execution_policy const& policy,
        HPX_ENUM_FWD_ARGS(N, Arg, arg)) const
    {
        try {
            return parallel::v1::detail::algorithm_result<
                    sequential_task_execution_policy, result_type
                >::get(Derived::sequential(
                    policy, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
        }
        catch (...) {
            return parallel::v1::detail::handle_exception<
                    sequential_task_execution_policy, result_type
                >::call();
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename parallel::v1::detail::algorithm_result<
        sequential_task_execution_policy, result_type
    >::type
    call(sequential_task_execution_policy const& policy,
        HPX_ENUM_FWD_ARGS(N, Arg, arg), boost::mpl::true_) const
    {
        try {
            hpx::future<result_type> result =
                hpx::async(get_async_fused_wrapper<Derived>(
                    hpx::util::forward_as_tuple(policy,
                        HPX_ENUM_FORWARD_ARGS(N, Arg, arg))
                ));

            return parallel::v1::detail::algorithm_result<
                    sequential_task_execution_policy, result_type
                >::get(std::move(result));
        }
        catch (...) {
            return parallel::v1::detail::handle_exception<
                    sequential_task_execution_policy, result_type
                >::call();
        }
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename parallel::v1::detail::algorithm_result<
        parallel_task_execution_policy, result_type
    >::type
    call(parallel_task_execution_policy const& policy,
        HPX_ENUM_FWD_ARGS(N, Arg, arg), boost::mpl::true_) const
    {
        try {
            return parallel::v1::detail::algorithm_result<
                    parallel_task_execution_policy, result_type
                >::get(Derived::sequential(
                    policy, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
        }
        catch (...) {
            return parallel::v1::detail::handle_exception<
                    parallel_task_execution_policy, result_type
                >::call();
        }
    }

    template <typename ExPolicy, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename parallel::v1::detail::algorithm_result<ExPolicy, result_type>::type
    call(ExPolicy const& policy, HPX_ENUM_FWD_ARGS(N, Arg, arg),
        boost::mpl::false_) const
    {
        return Derived::parallel(policy, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    result_type call(parallel::v1::execution_policy const& policy,
        HPX_ENUM_FWD_ARGS(N, Arg, arg), boost::mpl::false_) const
    {
        HPX_PARALLEL_DISPATCH(policy, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    result_type call(parallel::v1::execution_policy const& policy,
         HPX_ENUM_FWD_ARGS(N, Arg, arg), boost::mpl::true_) const
    {
        return call(seq, HPX_ENUM_FORWARD_ARGS(N, Arg, arg),
            boost::mpl::true_());
    }

#undef N

#endif
