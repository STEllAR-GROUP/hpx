//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_HANDLE_REMOTE_EXCEPTIONS_DEC_28_2014_0316PM)
#define HPX_PARALLEL_UTIL_DETAIL_HANDLE_REMOTE_EXCEPTIONS_DEC_28_2014_0316PM

#include <hpx/config.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/execution_policy.hpp>

#include <exception>
#include <list>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct handle_remote_exceptions
    {
        // std::bad_alloc has to be handled separately
        static void call(std::exception_ptr const& e,
            std::list<std::exception_ptr>& errors)
        {
            try {
                std::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba) {
                throw ba;
            }
            catch (exception_list const& el) {
                for (std::exception_ptr const& ex: el)
                    errors.push_back(ex);
            }
            catch (...) {
                errors.push_back(e);
            }
        }

        template <typename T>
        static void call(std::vector<hpx::future<T> > const& workitems,
            std::list<std::exception_ptr>& errors)
        {
            for (hpx::future<T> const& f: workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                throw exception_list(std::move(errors));
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T> > const& workitems,
            std::list<std::exception_ptr>& errors)
        {
            for (hpx::shared_future<T> const& f: workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                throw exception_list(std::move(errors));
        }
    };

    template <>
    struct handle_remote_exceptions<execution::parallel_unsequenced_policy>
    {
        HPX_NORETURN static void call(
            std::exception_ptr const&, std::list<std::exception_ptr>&)
        {
            hpx::terminate();
        }

        template <typename T>
        static void call(std::vector<hpx::future<T> > const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (hpx::future<T> const& f: workitems)
            {
                if (f.has_exception())
                    hpx::terminate();
            }
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T> > const& workitems,
            std::list<std::exception_ptr>&)
        {
            for (hpx::shared_future<T> const& f: workitems)
            {
                if (f.has_exception())
                    hpx::terminate();
            }
        }
    };
}}}}

#endif
