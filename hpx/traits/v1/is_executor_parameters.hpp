//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_PARAMETERS_V1_AUG_01_2015_0709AM)
#define HPX_TRAITS_IS_EXECUTOR_PARAMETERS_V1_AUG_01_2015_0709AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/traits/is_executor_parameters.hpp>

namespace hpx { namespace parallel { namespace execution { namespace detail
{
    template <typename T>
    struct is_executor_parameters;
}}}}

namespace hpx { namespace parallel { inline namespace v3
{
    ///////////////////////////////////////////////////////////////////////////
    /// 1. The type is_executor_parameters can be used to detect executor
    ///    parameters types for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined executor,
    ///    is_executor_parameters<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_executor_parameters is undefined.
    ///
    template <typename T>
    using is_executor_parameters = execution::detail::is_executor_parameters<T>;

    template <typename Executor, typename Enable = void>
    struct executor_parameter_traits;
}}}

#endif
#endif

