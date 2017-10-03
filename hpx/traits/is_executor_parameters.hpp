//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#if !defined(HPX_TRAITS_IS_EXECUTOR_PARAMETERS_MAY_19_2017_0232PM)
#define HPX_TRAITS_IS_EXECUTOR_PARAMETERS_MAY_19_2017_0232PM

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/traits/v1/is_executor_parameters.hpp>     // backwards compatibility

#include <functional>
#include <type_traits>

#include <boost/ref.hpp>

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Parameters, typename Enable = void>
    struct is_executor_parameters;
}}

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Default sequential executor parameters
    struct sequential_executor_parameters {};

    // If an executor exposes 'executor_parameter_type' this type is
    // assumed to represent the default parameters for the given executor
    // type.
    template <typename Executor, typename Enable = void>
    struct extract_executor_parameters
    {
        // by default, assume sequential execution
        using type = sequential_executor_parameters;
    };

    template <typename Executor>
    struct extract_executor_parameters<Executor,
        typename hpx::util::always_void<
            typename Executor::executor_parameters_type
        >::type>
    {
        using type = typename Executor::executor_parameters_type;
    };

    ///////////////////////////////////////////////////////////////////////
    // If a parameters type exposes 'has_variable_chunk_size' aliased to
    // std::true_type it is assumed that the number of loop iterations to
    // combine is different for each of the generated chunks.
    template <typename Parameters, typename Enable = void>
    struct extract_has_variable_chunk_size
    {
        // by default, assume equally sized chunks
        using type = std::false_type;
    };

    template <typename Parameters>
    struct extract_has_variable_chunk_size<Parameters,
        typename hpx::util::always_void<
            typename Parameters::has_variable_chunk_size
        >::type>
    {
        using type = typename Parameters::has_variable_chunk_size;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_executor_parameters
          : std::false_type
        {};

        template <>
        struct is_executor_parameters<sequential_executor_parameters>
          : std::true_type
        {};

        template <typename T>
        struct is_executor_parameters< ::std::reference_wrapper<T> >
          : hpx::traits::is_executor_parameters<T>
        {};

        template <typename T>
        struct is_executor_parameters< ::boost::reference_wrapper<T> >
          : hpx::traits::is_executor_parameters<T>
        {};
        /// \endcond
    }

    template <typename T>
    struct is_executor_parameters
      : detail::is_executor_parameters<typename std::decay<T>::type>
    {};

    template <typename T>
    using is_executor_parameters_t = typename is_executor_parameters<T>::type;

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool is_executor_parameters_v = is_executor_parameters<T>::value;
#endif
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Parameters, typename Enable>
    struct is_executor_parameters
      : parallel::execution::is_executor_parameters<
            typename std::decay<Parameters>::type>
    {};

    template <typename T>
    using is_executor_parameters_t = typename is_executor_parameters<T>::type;

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool is_executor_parameters_v = is_executor_parameters<T>::value;
#endif
}}

#endif

