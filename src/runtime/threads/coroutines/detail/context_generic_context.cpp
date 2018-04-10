//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/coroutines/detail/context_generic_context.hpp>

#include <cstdint>
#include <utility>

namespace hpx { namespace threads { namespace coroutines
{
    namespace detail { namespace generic_context
    {
        void fcontext_context_impl::reset_stack()
        {
            if (ctx_)
            {
#if defined(_POSIX_VERSION)
                void* limit = static_cast<char*>(stack_pointer_) - stack_size_;
                if(posix::reset_stack(limit, stack_size_))
                    increment_stack_unbind_count();
#else
                // nothing we can do here ...
#endif
            }
        }

        void fcontext_context_impl::rebind_stack()
        {
            if (ctx_)
            {
                increment_stack_recycle_count();
#if BOOST_VERSION < 106100
                ctx_ =
                    boost::context::make_fcontext(stack_pointer_, stack_size_, funp_);
#else
                ctx_ =
                    boost::context::detail::make_fcontext(
                            stack_pointer_, stack_size_, funp_);
#endif
            }
        }

        fcontext_context_impl::counter_type& fcontext_context_impl
            ::get_stack_unbind_counter()
        {
            static counter_type counter(0);
            return counter;
        }
        std::uint64_t fcontext_context_impl::get_stack_unbind_count(bool reset)
        {
            return util::get_and_reset_value(get_stack_unbind_counter(), reset);
        }
        std::uint64_t fcontext_context_impl::increment_stack_unbind_count()
        {
            return ++get_stack_unbind_counter();
        }

        fcontext_context_impl::counter_type& fcontext_context_impl
            ::get_stack_recycle_counter()
        {
            static counter_type counter(0);
            return counter;
        }
        std::uint64_t fcontext_context_impl::get_stack_recycle_count(bool reset)
        {
            return util::get_and_reset_value(get_stack_recycle_counter(), reset);
        }
        std::uint64_t fcontext_context_impl::increment_stack_recycle_count()
        {
            return ++get_stack_recycle_counter();
        }

    }}
}}}

#endif
