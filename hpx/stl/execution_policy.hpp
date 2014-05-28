//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM)
#define HPX_STL_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/decay.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <memory>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    /// The parallel_execution_policy is intended to specify the parallel
    /// execution policy for algorithms. The specific scheduling strategy will
    /// be chosen by the implementation depending on the algorithm being used.
    struct parallel_execution_policy {};

    /// Default parallel execution policy object.
    parallel_execution_policy const par;

    /// The sequential_execution_policy is intend to specify the sequential
    /// execution policy for algorithms.
    struct sequential_execution_policy {};

    /// Default sequential execution policy object.
    sequential_execution_policy const seq;

    /// The vector_execution_policy is intend to specify the vector execution
    /// policy for algorithms.
    struct vector_execution_policy {};

    /// Default vector execution policy object.
    vector_execution_policy const vec;

    ///////////////////////////////////////////////////////////////////////////
    /// The is_execution_policy is intended to test if specified type is of
    /// execution policy type.
    template <typename T>
    struct is_execution_policy
      : boost::mpl::false_
    {};

    template <>
    struct is_execution_policy<parallel_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_execution_policy<vector_execution_policy>
      : boost::mpl::true_
    {};

    template <>
    struct is_execution_policy<sequential_execution_policy>
      : boost::mpl::true_
    {};

    class execution_policy;

    template <>
    struct is_execution_policy<execution_policy>
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename ExPolicy, typename T>
        struct enable_if_parallel
          : public boost::enable_if<
                boost::is_base_of<
                    parallel_execution_policy, typename util::decay<ExPolicy>::type
                >::value ||
                boost::is_base_of<
                    vector_execution_policy, typename util::decay<ExPolicy>::type
                >::value
              , T>
        {};

        template <typename ExPolicy, typename T>
        struct enable_if_policy
          : public boost::enable_if<
                boost::is_base_of<
                    sequential_execution_policy, typename util::decay<ExPolicy>::type
                >::value ||
                boost::is_base_of<
                    parallel_execution_policy, typename util::decay<ExPolicy>::type
                >::value ||
                boost::is_base_of<
                    vector_execution_policy, typename util::decay<ExPolicy>::type
                >::value ||
                boost::is_base_of<
                    execution_policy, typename util::decay<ExPolicy>::type
                >::value
              , T>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The execution_policy is intended to specify the dynamic execution
    /// policy for algorithms.
    class execution_policy
    {
    private:
        std::shared_ptr<void> inner_;
        std::type_info const* type_;

    public:
        /// Constructs a new execution_policy object.
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        execution_policy(ExPolicy const& policy,
                typename detail::enable_if_policy<ExPolicy, ExPolicy>::type* = 0)
          : inner_(boost::make_shared<ExPolicy>(policy)),
            type_(&typeid(ExPolicy))
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Cannot assign dynamic execution policy.");
        }

        /// Move constructs a new execution_policy object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy(execution_policy&& policy)
          : inner_(std::move(policy.inner_)),
            type_(policy.type_)
        {
            policy.type_ = 0;
        }

        /// Assigns a new execution policy to the object.
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        typename detail::enable_if_policy<ExPolicy, execution_policy>::type&
        operator=(ExPolicy const& policy)
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Cannot assign dynamic execution policy.");

            inner_ = boost::make_shared<ExPolicy>(policy);
            type_ = &typeid(ExPolicy);
            return *this;
        }

        /// Move assigns a new execution policy to the object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy& operator=(execution_policy && policy)
        {
            if (this != &policy)
            {
                inner_ = std::move(policy.inner_);
                type_ = policy.type_;
                policy.type_ = 0;
            }
            return *this;
        }

        /// Returns the type_info of the inner policy.
        type_info const& type() const BOOST_NOEXCEPT
        {
            return *type_;
        }

        /// Returns the inner policy if type matches.
        template <typename ExPolicy>
        ExPolicy* get() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                !boost::is_same<ExPolicy, execution_policy>::value,
                "Incorrect execution policy parameter.");
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (*type_ != typeid(_ExPolicy))
                return 0;

            return static_cast<ExPolicy*>(inner_.get());
        }
    };
}}

#endif
