//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXCEPTION_LIST_JUN_25_2014_1055PM)
#define HPX_PARALLEL_EXCEPTION_LIST_JUN_25_2014_1055PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/parallel/execution_policy.hpp>

namespace hpx { namespace parallel
{
    namespace detail
    {
        template <typename ExPolicy>
        struct handle_exception
        {
            BOOST_ATTRIBUTE_NORETURN static void call()
            {
                try {
                    throw;
                }
                catch(std::bad_alloc const& e) {
                    boost::throw_exception(e);
                }
                catch (...) {
                    boost::throw_exception(
                        hpx::exception_list(boost::current_exception())
                    );
                }
            }
        };

        template <>
        struct handle_exception<vector_execution_policy>
        {
            BOOST_ATTRIBUTE_NORETURN static void call()
            {
                std::cout << "terminated";
                // any exceptions thrown by algorithms executed with the
                // vector_execution_policy are to call terminate.
                hpx::terminate();
            }
        };
    }

    // we're just reusing our existing implementation
    using hpx::exception_list;
}}

#endif
