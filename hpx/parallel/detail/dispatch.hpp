//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM)
#define HPX_PARALLEL_DISPATCH_JUN_25_2014_1145PM

#define HPX_PARALLEL_DISPATCH(policy, func, ...)                              \
    switch(detail::which(policy))                                             \
    {                                                                         \
    case detail::execution_policy_enum::sequential:                           \
        return func(*policy.get<sequential_execution_policy>(), __VA_ARGS__,  \
            boost::mpl::true_());                                             \
                                                                              \
    case detail::execution_policy_enum::parallel:                             \
        return func(*policy.get<parallel_execution_policy>(), __VA_ARGS__,    \
            boost::mpl::false_());                                            \
                                                                              \
    case detail::execution_policy_enum::vector:                               \
        return func(*policy.get<parallel_vector_execution_policy>(),          \
            __VA_ARGS__, boost::mpl::false_());                               \
                                                                              \
    case detail::execution_policy_enum::task:                                 \
        return func(par, __VA_ARGS__, boost::mpl::false_());                  \
                                                                              \
    default:                                                                  \
        HPX_THROW_EXCEPTION(hpx::bad_parameter,                               \
            "hpx::parallel::" BOOST_PP_STRINGIZE(func),                       \
            "Not supported execution policy");                                \
    }                                                                         \
    /**/

#endif
