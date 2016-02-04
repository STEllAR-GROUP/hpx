//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM)
#define HPX_PARALLEL_ALGORITHM_REMOTE_DISPATCH_OCT_15_2014_0938PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/colocating_distribution_policy.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct algorithm_result_helper
    {
        template <typename T_>
        static BOOST_FORCEINLINE T_ call(T_&& val)
        {
            return std::forward<T_>(val);
        }
    };

    template <>
    struct algorithm_result_helper<future<void> >
    {
        static BOOST_FORCEINLINE future<void> call(future<void>&& f)
        {
            return std::move(f);
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<
        Iterator,
        typename boost::enable_if<
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator
                    >::is_segmented_local_iterator
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator>
            traits;

        static BOOST_FORCEINLINE Iterator
        call(typename traits::local_raw_iterator&& it)
        {
            return traits::remote(std::move(it));
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<
        std::pair<Iterator1, Iterator2>,
        typename boost::enable_if<
            typename boost::mpl::or_<
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator1
                    >::is_segmented_local_iterator,
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator2
                    >::is_segmented_local_iterator
            >::type
        >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1>
            traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2>
            traits2;

        static BOOST_FORCEINLINE
        std::pair<
            typename traits1::local_iterator, typename traits2::local_iterator
        >
        call(std::pair<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator
            > && p)
        {
            return std::make_pair(
                traits1::remote(std::move(p.first)),
                traits2::remote(std::move(p.second)));
        }
    };

    template <typename Iterator>
    struct algorithm_result_helper<
        future<Iterator>,
        typename boost::enable_if<
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator
                    >::is_segmented_local_iterator
            >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator> traits;

        static BOOST_FORCEINLINE future<Iterator>
        call(future<typename traits::local_raw_iterator>&& f)
        {
            typedef future<typename traits::local_raw_iterator> argtype;
            return f.then(
                [](argtype&& f) -> Iterator
                {
                    return traits::remote(f.get());
                });
        }
    };

    template <typename Iterator1, typename Iterator2>
    struct algorithm_result_helper<
        future<std::pair<Iterator1, Iterator2> >,
        typename boost::enable_if<
            typename boost::mpl::or_<
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator1
                    >::is_segmented_local_iterator,
                typename hpx::traits::segmented_local_iterator_traits<
                        Iterator2
                    >::is_segmented_local_iterator
            >::type
        >::type>
    {
        typedef hpx::traits::segmented_local_iterator_traits<Iterator1>
            traits1;
        typedef hpx::traits::segmented_local_iterator_traits<Iterator2>
            traits2;

        typedef std::pair<
                typename traits1::local_raw_iterator,
                typename traits2::local_raw_iterator
            > arg_type;

        static BOOST_FORCEINLINE
        future<std::pair<
            typename traits1::local_iterator, typename traits2::local_iterator
        > >
        call(future<arg_type> && f)
        {
            return f.then(
                [](future<arg_type> && f)
                ->  std::pair<
                        typename traits1::local_iterator,
                        typename traits2::local_iterator
                    >
                {
                    auto p = f.get();
                    return std::make_pair(
                        traits1::remote(p.first),
                        traits2::remote(p.second));
                });
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher_helper
    {
        static BOOST_FORCEINLINE R sequential(Algo const& algo,
            ExPolicy const& policy, Args const&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result_helper<R>::call(
                    algo.call(policy, boost::mpl::true_(),
                        segmented_local_iterator_traits<Args>::local(args)...
                    )
                );
        }

        static BOOST_FORCEINLINE R parallel(Algo const& algo,
            ExPolicy const& policy, Args const&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            return
                detail::algorithm_result_helper<R>::call(
                    algo.call(policy, boost::mpl::false_(),
                        segmented_local_iterator_traits<Args>::local(args)...
                    )
                );
        }
    };

    template <typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher_helper<void, Algo, ExPolicy, Args...>
    {
        static BOOST_FORCEINLINE
        typename parallel::util::detail::algorithm_result<ExPolicy>::type
        sequential(Algo const& algo, ExPolicy const& policy, Args const&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            algo.call(policy, boost::mpl::true_(),
                    segmented_local_iterator_traits<Args>::local(args)...
                );
        }

        static BOOST_FORCEINLINE
        typename parallel::util::detail::algorithm_result<ExPolicy>::type
        parallel(Algo const& algo, ExPolicy const& policy, Args const&... args)
        {
            using hpx::traits::segmented_local_iterator_traits;
            algo.call(policy, boost::mpl::false_(),
                    segmented_local_iterator_traits<Args>::local(args)...
                );
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename... Args>
    struct dispatcher
    {
        typedef typename parallel::util::detail::algorithm_result<
                ExPolicy, typename hpx::util::decay<Algo>::type::result_type
            >::type result_type;

        typedef dispatcher_helper<
                result_type, Algo, ExPolicy, Args...
            > base_dispatcher;

        static BOOST_FORCEINLINE result_type sequential(Algo const& algo,
            ExPolicy const& policy, Args const&... args)
        {
            return base_dispatcher::sequential(algo, policy, args...);
        }

        static BOOST_FORCEINLINE result_type parallel(Algo const& algo,
            ExPolicy const& policy, Args const&... args)
        {
            return base_dispatcher::sequential(algo, policy, args...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename F>
    struct algorithm_invoker_action;

    // sequential
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::true_, R(Args const& ...)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&, Args const&...),
            &dispatcher<Algo, ExPolicy, Args...>::sequential,
            algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::true_, R(Args const& ...)>
        >::type
    {};

    // parallel
    template <typename Algo, typename ExPolicy, typename R, typename... Args>
    struct algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::false_, R(Args const& ...)>
        : hpx::actions::make_action<
            R (*)(Algo const&, ExPolicy const&, Args const&...),
            &dispatcher<Algo, ExPolicy, Args...>::parallel,
            algorithm_invoker_action<
                Algo, ExPolicy, boost::mpl::false_, R(Args const& ...)>
        >::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    BOOST_FORCEINLINE
    future<typename hpx::util::decay<Algo>::type::result_type>
    dispatch_async(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq, Args&&... args)
    {
        typedef typename hpx::util::decay<Algo>::type algo_type;
        typedef typename parallel::util::detail::algorithm_result<
                ExPolicy, typename algo_type::result_type
            >::type result_type;

        algorithm_invoker_action<
            algo_type, ExPolicy, IsSeq,
            result_type(typename hpx::util::decay<Args>::type const&...)
        > act;

        return hpx::async(act, hpx::colocated(id), std::forward<Algo>(algo),
            policy, std::forward<Args>(args)...);
    }

    template <typename Algo, typename ExPolicy, typename IsSeq, typename... Args>
    BOOST_FORCEINLINE
    typename hpx::util::decay<Algo>::type::result_type
    dispatch(id_type const& id, Algo && algo, ExPolicy const& policy,
        IsSeq is_seq, Args&&... args)
    {
        // synchronously invoke remote operation
        future<typename hpx::util::decay<Algo>::type::result_type> f =
            dispatch_async(id, std::forward<Algo>(algo), policy, is_seq,
                std::forward<Args>(args)...);
        f.wait();

        // handle any remote exceptions
        if (f.has_exception())
        {
            std::list<boost::exception_ptr> errors;
            parallel::util::detail::handle_remote_exceptions<
                    ExPolicy
                >::call(f.get_exception_ptr(), errors);

            HPX_ASSERT(errors.empty());
            boost::throw_exception(exception_list(std::move(errors)));
        }
        return f.get();
    }
}}}}

#endif
