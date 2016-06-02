//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_PARAMETERS_AUG_01_2015_0709AM)
#define HPX_TRAITS_IS_EXECUTOR_PARAMETERS_AUG_01_2015_0709AM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>
#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
#include <functional>
#endif

#include <boost/ref.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct executor_parameters_tag {};

    struct executor_parameters_chunk_tag : executor_parameters_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_executor_parameters
          : std::is_base_of<executor_parameters_tag, T>
        {};

        template <>
        struct is_executor_parameters<executor_parameters_tag>
          : std::false_type
        {};

        template <typename T>
        struct is_executor_parameters< ::boost::reference_wrapper<T> >
          : is_executor_parameters<typename hpx::util::decay<T>::type>
        {};

#if defined(HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)
        template <typename T>
        struct is_executor_parameters< ::std::reference_wrapper<T> >
          : is_executor_parameters<typename hpx::util::decay<T>::type>
        {};
#endif
        /// \endcond
    }

    template <typename T>
    struct is_executor_parameters
      : detail::is_executor_parameters<typename hpx::util::decay<T>::type>
    {};

    template <typename Executor, typename Enable = void>
    struct executor_parameter_traits;
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Parameters, typename Enable>
    struct is_executor_parameters
      : parallel::v3::is_executor_parameters<Parameters>
    {};
}}

#endif

