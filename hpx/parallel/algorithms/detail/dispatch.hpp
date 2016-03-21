//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM)
#define HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <boost/mpl/bool.hpp>

#include <string>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct local_algorithm_result
    {
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result
            >::local_raw_iterator type;
    };

    template <typename Result1, typename Result2>
    struct local_algorithm_result<std::pair<Result1, Result2> >
    {
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result1
            >::local_raw_iterator type1;
        typedef typename hpx::traits::segmented_local_iterator_traits<
                Result2
            >::local_raw_iterator type2;

        typedef std::pair<type1, type2> type;
    };

    template <>
    struct local_algorithm_result<void>
    {
        typedef void type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Result = void>
    struct algorithm
    {
    private:
        Derived const& derived() const
        {
            return static_cast<Derived const&>(*this);
        }

    public:
        typedef Result result_type;
        typedef typename local_algorithm_result<result_type>::type
            local_result_type;

        explicit algorithm(char const* const name)
          : name_(name)
        {}

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call(ExPolicy && policy, boost::mpl::true_, Args&&... args) const
        {
            try {
                return parallel::util::detail::algorithm_result<
                        ExPolicy, local_result_type
                    >::get(Derived::sequential(std::forward<ExPolicy>(policy),
                        std::forward<Args>(args)...));
            }
            catch (...) {
                detail::handle_exception<
                    typename hpx::util::decay<ExPolicy>::type,
                    local_result_type
                >::call();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        operator()(ExPolicy && policy, Args&&... args) const
        {
            try {
                return parallel::util::detail::algorithm_result<
                        ExPolicy, local_result_type
                    >::get(Derived::sequential(std::forward<ExPolicy>(policy),
                        std::forward<Args>(args)...));
            }
            catch (...) {
                return detail::handle_exception<
                        typename hpx::util::decay<ExPolicy>::type,
                        local_result_type
                    >::call();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call_sequential(ExPolicy && policy, Args&&... args) const
        {
            try {
                hpx::future<local_result_type> result =
                    hpx::async(derived(), std::forward<ExPolicy>(policy),
                        std::forward<Args>(args)...);

                return parallel::util::detail::algorithm_result<
                        ExPolicy, local_result_type
                    >::get(std::move(result));
            }
            catch (...) {
                return detail::handle_exception<
                        typename hpx::util::decay<ExPolicy>::type,
                        local_result_type
                    >::call();
            }
        }


        template <typename... Args>
        typename parallel::util::detail::algorithm_result<
            sequential_task_execution_policy, local_result_type
        >::type
        call(sequential_task_execution_policy policy, boost::mpl::true_,
            Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            sequential_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(sequential_task_execution_policy_shim<Executor, Parameters>& policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            sequential_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(sequential_task_execution_policy_shim<Executor, Parameters> && policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            sequential_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(sequential_task_execution_policy_shim<Executor, Parameters> const& policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Args>
        typename parallel::util::detail::algorithm_result<
            parallel_task_execution_policy, local_result_type
        >::type
        call(parallel_task_execution_policy policy, boost::mpl::true_,
            Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            parallel_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(parallel_task_execution_policy_shim<Executor, Parameters>& policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            parallel_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(parallel_task_execution_policy_shim<Executor, Parameters> && policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename Executor, typename Parameters, typename... Args>
        typename parallel::util::detail::algorithm_result<
            parallel_task_execution_policy_shim<Executor, Parameters>,
            local_result_type
        >::type
        call(parallel_task_execution_policy_shim<Executor, Parameters> const& policy,
            boost::mpl::true_, Args&&... args) const
        {
            return call_sequential(policy, std::forward<Args>(args)...);
        }

        template <typename ExPolicy, typename... Args>
        typename parallel::util::detail::algorithm_result<
            ExPolicy, local_result_type
        >::type
        call(ExPolicy && policy, boost::mpl::false_, Args&&... args) const
        {
            return Derived::parallel(std::forward<ExPolicy>(policy),
                std::forward<Args>(args)...);
        }

        ///////////////////////////////////////////////////////////////////////////
        template <typename... Args>
        local_result_type
        call(parallel::execution_policy policy, boost::mpl::false_,
            Args&&... args) const
        {
            // this implementation is not nice, however we don't have variadic
            // virtual functions accepting template arguments and supporting
            // perfect forwarding
            std::type_info const& t = policy.type();

            if (t == typeid(sequential_execution_policy))
            {
                return call(*policy.get<sequential_execution_policy>(),
                    boost::mpl::true_(), std::forward<Args>(args)...);
            }

            if (t == typeid(sequential_task_execution_policy))
            {
                return call(seq, boost::mpl::true_(),
                    std::forward<Args>(args)...);
            }

            if (t == typeid(parallel_execution_policy))
            {
                return call(*policy.get<parallel_execution_policy>(),
                    boost::mpl::false_(), std::forward<Args>(args)...);
            }

            if (t == typeid(parallel_task_execution_policy))
            {
                parallel_task_execution_policy const& t =
                    *policy.get<parallel_task_execution_policy>();

                return call(par.with(t.parameters()),
                    boost::mpl::false_(), std::forward<Args>(args)...);
            }

            if (t == typeid(parallel_vector_execution_policy))
            {
                return call(*policy.get<parallel_vector_execution_policy>(),
                    boost::mpl::false_(), std::forward<Args>(args)...);
            }

            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                std::string("hpx::parallel::") + name_,
                "The given execution policy is not supported");
        }

        template <typename... Args>
        local_result_type
        call(parallel::execution_policy, boost::mpl::true_, Args&&... args) const
        {
            return call(seq, boost::mpl::true_(), std::forward<Args>(args)...);
        }

    private:
        char const* const name_;

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive&, unsigned int)
        {
            // no need to serialize 'name_' as it is always initialized by the
            // constructor
        }
    };
}}}}

#endif
