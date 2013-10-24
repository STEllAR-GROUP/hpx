//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_COMPONENTS_DISTRIBUTED_NEW_HPP
#define HPX_COMPONENTS_DISTRIBUTED_NEW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/remote_object/new.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/components/remote_object/preprocessed/distributed_new.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/distributed_new_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/components/remote_object/distributed_new.hpp>                  \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace components {
#if N == 0
    template <typename T>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count)
#else
    template <typename T, BOOST_PP_ENUM_PARAMS(N, typename A)>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, HPX_ENUM_FWD_ARGS(N, A, a))
#endif
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();

        std::vector<naming::id_type> prefixes = find_all_localities(type);

        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();

        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();

        std::vector<lcos::future<object<T> > > res;

        res.reserve(count);

        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;

            if (excess != 0) {
                --excess;
                ++numcreate;
            }

            if (created_count + numcreate > count)
                numcreate = count - created_count;

            if (numcreate == 0)
                break;

            for (std::size_t i = 0; i < numcreate; ++i) {
#if N == 0
                res.push_back(
                    new_<T>(prefix)
                );
#else
                res.push_back(
                    new_<T>(prefix, HPX_ENUM_FORWARD_ARGS(N, A, a))
                );
#endif
            }

            created_count += numcreate;
            if (created_count >= count)
                break;
        }

        return res;
    }
}}

#undef N

#endif
