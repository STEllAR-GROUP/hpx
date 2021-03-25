//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/distribution_policies/colocating_distribution_policy.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <exception>
#include <list>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct algorithm_result_helper
    {
        template <typename T_>
        HPX_FORCEINLINE static constexpr T_ call(T_&& val)
        {
            return std::forward<T_>(val);
        }
    };

    template <>
    struct algorithm_result_helper<future<void>>
    {
        HPX_FORCEINLINE static future<void> call(future<void>&& f)
        {
            return std::move(f);
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<Iterator,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator>::value>::type>
    {
        using traits = hpx::traits::segmented_local_iterator_traits<Iterator>;

        HPX_FORCEINLINE static Iterator call(
            typename traits::local_raw_iterator&& it)
        {
            return traits::remote(std::move(it));
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<std::pair<Iterator1, Iterator2>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;

        HPX_FORCEINLINE static std::pair<typename traits1::local_iterator,
            typename traits2::local_iterator>
        call(std::pair<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator>&& p)
        {
            return std::make_pair(traits1::remote(std::move(p.first)),
                traits2::remote(std::move(p.second)));
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<util::in_out_result<Iterator1, Iterator2>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;

        HPX_FORCEINLINE static util::in_out_result<
            typename traits1::local_iterator, typename traits2::local_iterator>
        call(util::in_out_result<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator>&& p)
        {
            return util::in_out_result<typename traits1::local_iterator,
                typename traits2::local_iterator>{
                traits1::remote(std::move(p.in)),
                traits2::remote(std::move(p.out))};
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<hpx::tuple<Iterator1, Iterator2, Iterator3>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator3>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;
        using traits3 = hpx::traits::segmented_local_iterator_traits<Iterator3>;

        HPX_FORCEINLINE static hpx::tuple<typename traits1::local_iterator,
            typename traits2::local_iterator, typename traits3::local_iterator>
        call(hpx::tuple<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator,
            typename traits3::local_raw_iterator>&& p)
        {
            return hpx::make_tuple(traits1::remote(std::move(hpx::get<0>(p))),
                traits2::remote(std::move(hpx::get<1>(p))),
                traits3::remote(std::move(hpx::get<2>(p))));
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<
        util::in_in_out_result<Iterator1, Iterator2, Iterator3>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator3>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;
        using traits3 = hpx::traits::segmented_local_iterator_traits<Iterator3>;

        HPX_FORCEINLINE static util::in_in_out_result<
            typename traits1::local_iterator, typename traits2::local_iterator,
            typename traits3::local_iterator>
        call(util::in_in_out_result<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator,
            typename traits3::local_raw_iterator>&& p)
        {
            return util::in_in_out_result<typename traits1::local_iterator,
                typename traits2::local_iterator,
                typename traits3::local_iterator>{
                traits1::remote(std::move(p.in1)),
                traits2::remote(std::move(p.in2)),
                traits3::remote(std::move(p.out))};
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<future<Iterator>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator>::value>::type>
    {
        using traits = hpx::traits::segmented_local_iterator_traits<Iterator>;

        HPX_FORCEINLINE static future<Iterator> call(
            future<typename traits::local_raw_iterator>&& f)
        {
            using argtype = future<typename traits::local_raw_iterator>;
            return f.then(hpx::launch::sync, [](argtype&& f) -> Iterator {
                return traits::remote(f.get());
            });
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<future<std::pair<Iterator1, Iterator2>>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;

        using arg_type = std::pair<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator>;

        HPX_FORCEINLINE static future<std::pair<
            typename traits1::local_iterator, typename traits2::local_iterator>>
        call(future<arg_type>&& f)
        {
            // different versions of clang-format produce different results
            // clang-format off
            return f.then(hpx::launch::sync,
                [](future<arg_type>&& f)
                    -> std::pair<typename traits1::local_iterator,
                        typename traits2::local_iterator> {
                    auto&& p = f.get();
                    return std::make_pair(
                        traits1::remote(p.first), traits2::remote(p.second));
                });
            // clang-format on
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<
        future<util::in_out_result<Iterator1, Iterator2>>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;

        using arg_type =
            util::in_out_result<typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator>;

        HPX_FORCEINLINE static future<util::in_out_result<
            typename traits1::local_iterator, typename traits2::local_iterator>>
        call(future<arg_type>&& f)
        {
            // different versions of clang-format produce different results
            // clang-format off
            return f.then(hpx::launch::sync,
                [](future<arg_type>&& f)
                    -> util::in_out_result<typename traits1::local_iterator,
                        typename traits2::local_iterator> {
                    auto&& p = f.get();
                    return util::in_out_result<typename traits1::local_iterator,
                        typename traits2::local_iterator>{traits1::remote(p.in),
                            traits2::remote(p.out)};
                });
            // clang-format on
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<
        future<hpx::tuple<Iterator1, Iterator2, Iterator3>>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator3>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;
        using traits3 = hpx::traits::segmented_local_iterator_traits<Iterator3>;

        using arg_type = hpx::tuple<typename traits1::local_raw_iterator,
            typename traits2::local_raw_iterator,
            typename traits3::local_raw_iterator>;

        HPX_FORCEINLINE static future<hpx::tuple<
            typename traits1::local_iterator, typename traits2::local_iterator,
            typename traits3::local_iterator>>
        call(future<arg_type>&& f)
        {
            // different versions of clang-format produce different results
            // clang-format off
            return f.then(hpx::launch::sync,
                [](future<arg_type>&& f)
                    -> hpx::tuple<typename traits1::local_iterator,
                        typename traits2::local_iterator,
                        typename traits3::local_iterator> {
                    auto&& p = f.get();
                    return hpx::make_tuple(
                        traits1::remote(std::move(hpx::get<0>(p))),
                        traits2::remote(std::move(hpx::get<1>(p))),
                        traits3::remote(std::move(hpx::get<2>(p))));
                });
            // clang-format on
        }
    };

    template <typename Iterator1, typename Iterator2, typename Iterator3>
    struct algorithm_result_helper<
        future<util::in_in_out_result<Iterator1, Iterator2, Iterator3>>,
        typename std::enable_if<
            hpx::traits::is_segmented_local_iterator<Iterator1>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator2>::value ||
            hpx::traits::is_segmented_local_iterator<Iterator3>::value>::type>
    {
        using traits1 = hpx::traits::segmented_local_iterator_traits<Iterator1>;
        using traits2 = hpx::traits::segmented_local_iterator_traits<Iterator2>;
        using traits3 = hpx::traits::segmented_local_iterator_traits<Iterator3>;

        using arg_type =
            util::in_in_out_result<typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator,
                typename traits3::local_raw_iterator>;

        HPX_FORCEINLINE static future<util::in_in_out_result<
            typename traits1::local_iterator, typename traits2::local_iterator,
            typename traits3::local_iterator>>
        call(future<arg_type>&& f)
        {
            // different versions of clang-format produce different results
            // clang-format off
            return f.then(hpx::launch::sync,
                [](future<arg_type>&& f)
                    -> util::in_in_out_result<typename traits1::local_iterator,
                        typename traits2::local_iterator,
                        typename traits3::local_iterator> {
                    auto&& p = f.get();
                    return  util::in_in_out_result<typename traits1::local_iterator,
                        typename traits2::local_iterator,
                        typename traits3::local_iterator>{
                        traits1::remote(std::move(p.in1)),
                        traits2::remote(std::move(p.in2)),
                        traits3::remote(std::move(p.out))};
                });
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename Algo>
    struct dispatcher_helper
    {
        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE static R sequential(
            Algo const& algo, ExPolicy&& policy, Args&&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return detail::algorithm_result_helper<R>::call(
                algo.call2(std::forward<ExPolicy>(policy), std::true_type(),
                    segmented_local_iterator_traits<std::decay_t<Args>>::local(
                        std::forward<Args>(args))...));
        }

        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE static R parallel(
            Algo const& algo, ExPolicy&& policy, Args&&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return detail::algorithm_result_helper<R>::call(
                algo.call2(std::forward<ExPolicy>(policy), std::false_type(),
                    segmented_local_iterator_traits<std::decay_t<Args>>::local(
                        std::forward<Args>(args))...));
        }
    };

    template <typename Algo>
    struct dispatcher_helper<void, Algo>
    {
        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE static
            typename parallel::util::detail::algorithm_result<ExPolicy>::type
            sequential(Algo const& algo, ExPolicy&& policy, Args&&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return algo.call2(std::forward<ExPolicy>(policy), std::true_type(),
                segmented_local_iterator_traits<std::decay_t<Args>>::local(
                    std::forward<Args>(args))...);
        }

        template <typename ExPolicy, typename... Args>
        HPX_FORCEINLINE static
            typename parallel::util::detail::algorithm_result<ExPolicy>::type
            parallel(Algo const& algo, ExPolicy&& policy, Args&&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return algo.call2(std::forward<ExPolicy>(policy), std::false_type(),
                segmented_local_iterator_traits<std::decay_t<Args>>::local(
                    std::forward<Args>(args))...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher
    {
        using result_type =
            typename parallel::util::detail::algorithm_result<ExPolicy,
                typename std::decay_t<Algo>::result_type>::type;

        using base_dispatcher = dispatcher_helper<result_type, Algo>;

        HPX_FORCEINLINE static result_type sequential(
            Algo const& algo, ExPolicy policy, Args... args)
        {
            return base_dispatcher::sequential(
                algo, std::move(policy), std::move(args)...);
        }

        HPX_FORCEINLINE static result_type parallel(
            Algo const& algo, ExPolicy policy, Args... args)
        {
            return base_dispatcher::parallel(
                algo, std::move(policy), std::move(args)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename F>
    struct algorithm_invoker_action;

    // sequential
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<Algo, ExPolicy, std::true_type, R(Args...)>
      : hpx::actions::make_action<R (*)(Algo const&, ExPolicy, Args...),
            &dispatcher<Algo, ExPolicy, Args...>::sequential,
            algorithm_invoker_action<Algo, ExPolicy, std::true_type,
                R(Args...)>>::type
    {
    };

    // parallel
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<Algo, ExPolicy, std::false_type, R(Args...)>
      : hpx::actions::make_action<R (*)(Algo const&, ExPolicy, Args...),
            &dispatcher<Algo, ExPolicy, Args...>::parallel,
            algorithm_invoker_action<Algo, ExPolicy, std::false_type,
                R(Args...)>>::type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq,
        typename... Args>
    HPX_FORCEINLINE future<typename std::decay_t<Algo>::result_type>
    dispatch_async(id_type const& id, Algo&& algo, ExPolicy const& policy,
        IsSeq, Args&&... args)
    {
        using algo_type = std::decay_t<Algo>;
        using result_type =
            typename parallel::util::detail::algorithm_result<ExPolicy,
                typename algo_type::result_type>::type;

        algorithm_invoker_action<algo_type, ExPolicy, typename IsSeq::type,
            result_type(std::decay_t<Args>...)>
            act;

        return hpx::async(act, hpx::colocated(id), std::forward<Algo>(algo),
            policy, std::forward<Args>(args)...);
    }

    template <typename Algo, typename ExPolicy, typename IsSeq,
        typename... Args>
    HPX_FORCEINLINE typename std::decay_t<Algo>::result_type dispatch(
        id_type const& id, Algo&& algo, ExPolicy const& policy, IsSeq is_seq,
        Args&&... args)
    {
        // synchronously invoke remote operation
        future<typename std::decay_t<Algo>::result_type> f =
            dispatch_async(id, std::forward<Algo>(algo), policy, is_seq,
                std::forward<Args>(args)...);
        f.wait();

        // handle any remote exceptions
        if (f.has_exception())
        {
            std::list<std::exception_ptr> errors;
            parallel::util::detail::handle_remote_exceptions<ExPolicy>::call(
                f.get_exception_ptr(),
                errors);    // NOLINT(bugprone-use-after-move)

            // NOLINTNEXTLINE(bugprone-use-after-move)
            HPX_ASSERT(errors.empty());
            throw exception_list(std::move(errors));
        }
        return f.get();
    }
}}}}    // namespace hpx::parallel::v1::detail
