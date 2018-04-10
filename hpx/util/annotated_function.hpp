//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ANNOTATED_FUNCTION_JAN_31_2017_1148AM)
#define HPX_ANNOTATED_FUNCTION_JAN_31_2017_1148AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/thread_description.hpp>

#if HPX_HAVE_ITTNOTIFY != 0
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/util/itt_notify.hpp>
#elif defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name) {}
        template <typename F>
        explicit HPX_HOST_DEVICE annotate_function(F && f) {}

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~annotate_function() {}
    };
#elif HPX_HAVE_ITTNOTIFY != 0
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name)
          : task_(thread_domain_,
                hpx::util::itt::string_handle(name))
        {}
        template <typename F>
        explicit annotate_function(F && f)
          : task_(thread_domain_,
                hpx::traits::get_function_annotation_itt<
                    typename std::decay<F>::type
                >::call(f))
        {}

    private:
        hpx::util::itt::thread_domain thread_domain_;
        hpx::util::itt::task task_;
    };
#elif defined(HPX_HAVE_APEX)
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name)
        {
            threads::set_self_apex_data(
                apex_update_task(threads::get_self_apex_data(),
                name));
        }
        template <typename F>
        explicit annotate_function(F && f)
        {
            threads::set_self_apex_data(
                apex_update_task(threads::get_self_apex_data(),
                hpx::util::thread_description(f)));
        }
    };
#else
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name)
          : desc_(hpx::threads::get_self_ptr() ?
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(), name) :
                nullptr)
        {}

        template <typename F>
        explicit annotate_function(F && f)
          : desc_(hpx::threads::get_self_ptr() ?
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(),
                    hpx::util::thread_description(f)) :
                nullptr)
        {}

        ~annotate_function()
        {
            if (hpx::threads::get_self_ptr())
            {
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(), desc_);
            }
        }

        hpx::util::thread_description desc_;
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F>
        struct annotated_function
        {
            annotated_function() noexcept
              : name_(nullptr)
            {}

            annotated_function(F const& f, char const* name)
              : f_(f), name_(name)
            {}

            annotated_function(F && f, char const* name)
              : f_(std::move(f)), name_(name)
            {}

        public:
            template <typename ... Ts>
            typename invoke_result<
                typename util::decay_unwrap<F>::type, Ts...>::type
            operator()(Ts && ... ts)
            {
                annotate_function func(name_);
                return util::invoke(f_, std::forward<Ts>(ts)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & f_;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<
                        typename util::decay_unwrap<F>::type
                    >::call(f_);
            }

            char const* get_function_annotation() const noexcept
            {
                return name_ ? name_ : typeid(f_).name();
            }

        private:
            typename util::decay_unwrap<F>::type f_;
            char const* name_;
        };
    }

    template <typename F>
    detail::annotated_function<typename std::decay<F>::type>
    annotated_function(F && f, char const* name = nullptr)
    {
        typedef detail::annotated_function<
            typename std::decay<F>::type
        > result_type;

        return result_type(std::forward<F>(f), name);
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    struct annotate_function
    {
        HPX_NON_COPYABLE(annotate_function);

        explicit annotate_function(char const* name) {}
        template <typename F>
        explicit HPX_HOST_DEVICE annotate_function(F && f) {}

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~annotate_function() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    F && annotated_function(F && f, char const* = nullptr)
    {
        return std::forward<F>(f);
    }
#endif
}}

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits
{
    template <typename F>
    struct get_function_address<util::detail::annotated_function<F> >
    {
        static std::size_t
        call(util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<util::detail::annotated_function<F> >
    {
        static char const*
        call(util::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
}}
#endif

#endif
