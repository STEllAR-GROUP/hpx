//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_HANDLE_LOCAL_EXCEPTIONS_OCT_03_2014_0142PM)
#define HPX_PARALLEL_UTIL_DETAIL_HANDLE_LOCAL_EXCEPTIONS_OCT_03_2014_0142PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/parallel/execution_policy.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct handle_local_exceptions
    {
        // std::bad_alloc has to be handled separately
        static void call(boost::exception_ptr const& e,
            std::list<boost::exception_ptr>& errors)
        {
            try {
                boost::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                errors.push_back(e);
            }
        }

        template <typename T>
        static void call(std::vector<hpx::future<T> > const& workitems,
            std::list<boost::exception_ptr>& errors)
        {
            for (hpx::future<T> const& f: workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                boost::throw_exception(exception_list(std::move(errors)));
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T> > const& workitems,
            std::list<boost::exception_ptr>& errors)
        {
            for (hpx::shared_future<T> const& f: workitems)
            {
                if (f.has_exception())
                    call(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                boost::throw_exception(exception_list(std::move(errors)));
        }

        template <typename T, typename Cleanup>
        static void call(std::vector<hpx::future<T> >& workitems,
            std::list<boost::exception_ptr>& errors, Cleanup && cleanup)
        {
            bool has_exception = false;
            boost::exception_ptr bad_alloc_exception;
            for (hpx::future<T>& f: workitems)
            {
                if (f.has_exception())
                {
                    boost::exception_ptr e = f.get_exception_ptr();
                    try {
                        boost::rethrow_exception(e);
                    }
                    catch (std::bad_alloc const&) {
                        bad_alloc_exception = e;
                    }
                    catch (...) {
                        errors.push_back(e);
                    }
                    has_exception = true;
                }
            }

            // If at least one partition failed with an exception, call
            // the cleanup function for all others (the failed partitioned
            // are assumed to have already run the cleanup).
            if (has_exception)
            {
                for (hpx::future<T>& f: workitems)
                {
                    if (!f.has_exception())
                        cleanup(f.get());
                }
            }

            if (bad_alloc_exception)
                boost::rethrow_exception(bad_alloc_exception);

            if (!errors.empty())
                boost::throw_exception(exception_list(std::move(errors)));
        }
    };

    template <>
    struct handle_local_exceptions<parallel_vector_execution_policy>
    {
        HPX_ATTRIBUTE_NORETURN static void call(
            boost::exception_ptr const&, std::list<boost::exception_ptr>&)
        {
            hpx::terminate();
        }

        template <typename T>
        static void call(std::vector<hpx::future<T> > const& workitems,
            std::list<boost::exception_ptr>&)
        {
            for (hpx::future<T> const& f: workitems)
            {
                if (f.has_exception())
                    hpx::terminate();
            }
        }

        template <typename T>
        static void call(std::vector<hpx::shared_future<T> > const& workitems,
            std::list<boost::exception_ptr>&)
        {
            for (hpx::shared_future<T> const& f: workitems)
            {
                if (f.has_exception())
                    hpx::terminate();
            }
        }

        template <typename T, typename Cleanup>
        static void call(std::vector<hpx::future<T> > const& workitems,
            std::list<boost::exception_ptr>&, Cleanup &&)
        {
            for (hpx::future<T> const& f: workitems)
            {
                if (f.has_exception())
                    hpx::terminate();
            }
        }
    };
}}}}

#endif
