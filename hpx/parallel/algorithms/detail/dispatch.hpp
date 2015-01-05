//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

#include <boost/mpl/bool.hpp>
#include <boost/serialization/serialization.hpp>

#include <string>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
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

        explicit algorithm(char const* const name)
          : name_(name)
        {}

        template <typename ExPolicy, typename... Args>
        typename parallel::v1::detail::algorithm_result<
            ExPolicy, result_type
        >::type
        call(ExPolicy const& policy, boost::mpl::true_, Args&&... args) const
        {
            try {
                return parallel::v1::detail::algorithm_result<
                        ExPolicy, result_type
                    >::get(Derived::sequential(policy, std::forward<Args>(args)...));
            }
            catch (...) {
                parallel::v1::detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename... Args>
        typename parallel::v1::detail::algorithm_result<
            sequential_task_execution_policy, result_type
        >::type
        operator()(sequential_task_execution_policy const& policy,
            Args&&... args) const
        {
            try {
                return parallel::v1::detail::algorithm_result<
                        sequential_task_execution_policy, result_type
                    >::get(Derived::sequential(policy, std::forward<Args>(args)...));
            }
            catch (...) {
                return parallel::v1::detail::handle_exception<
                        sequential_task_execution_policy, result_type
                    >::call();
            }
        }

        template <typename... Args>
        typename parallel::v1::detail::algorithm_result<
            sequential_task_execution_policy, result_type
        >::type
        call(sequential_task_execution_policy const& policy, boost::mpl::true_,
            Args&&... args) const
        {
            try {
                hpx::future<result_type> result =
                    hpx::async(derived(), policy, std::forward<Args>(args)...);

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

        template <typename... Args>
        typename parallel::v1::detail::algorithm_result<
            parallel_task_execution_policy, result_type
        >::type
        call(parallel_task_execution_policy const& policy, boost::mpl::true_,
            Args&&... args) const
        {
            try {
                return parallel::v1::detail::algorithm_result<
                        parallel_task_execution_policy, result_type
                    >::get(Derived::sequential(policy, std::forward<Args>(args)...));
            }
            catch (...) {
                return parallel::v1::detail::handle_exception<
                        parallel_task_execution_policy, result_type
                    >::call();
            }
        }

        template <typename ExPolicy, typename... Args>
        typename parallel::v1::detail::algorithm_result<ExPolicy, result_type>::type
        call(ExPolicy const& policy, boost::mpl::false_, Args&&... args) const
        {
            return Derived::parallel(policy, std::forward<Args>(args)...);
        }

        ///////////////////////////////////////////////////////////////////////////
        template <typename... Args>
        result_type
        call(parallel::v1::execution_policy const& policy, boost::mpl::false_,
            Args&&... args) const
        {
            switch(detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return call(*policy.get<sequential_execution_policy>(),
                    boost::mpl::true_(), std::forward<Args>(args)...);

            case detail::execution_policy_enum::sequential_task:
                return call(seq, boost::mpl::true_(), std::forward<Args>(args)...);

            case detail::execution_policy_enum::parallel:
                return call(*policy.get<parallel_execution_policy>(),
                    boost::mpl::false_(), std::forward<Args>(args)...);

            case detail::execution_policy_enum::parallel_task:
                {
                    parallel_task_execution_policy const& t =
                        *policy.get<parallel_task_execution_policy>();
                    return call(par(t.get_executor(), t.get_chunk_size()),
                        boost::mpl::false_(), std::forward<Args>(args)...);
                }

            case detail::execution_policy_enum::parallel_vector:
                return call(*policy.get<parallel_vector_execution_policy>(),
                    boost::mpl::false_(), std::forward<Args>(args)...);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    std::string("hpx::parallel::") + name_,
                    "Not supported execution policy");
            }
        }

        template <typename... Args>
        result_type
        call(parallel::v1::execution_policy const& policy, boost::mpl::true_,
            Args&&... args) const
        {
            return call(seq, boost::mpl::true_(), std::forward<Args>(args)...);
        }

    private:
        char const* const name_;

        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int)
        {
        }
    };
}}}}

#endif
