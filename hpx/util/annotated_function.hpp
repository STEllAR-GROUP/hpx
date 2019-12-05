//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ANNOTATED_FUNCTION_JAN_31_2017_1148AM)
#define HPX_ANNOTATED_FUNCTION_JAN_31_2017_1148AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/functional/invoke.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/util/thread_description.hpp>

#if HPX_HAVE_ITTNOTIFY != 0
#include <hpx/concurrency/itt_notify.hpp>
#elif defined(HPX_HAVE_APEX)
#include <hpx/util/external_timer.hpp>
#endif
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <string>

namespace hpx { namespace util
{
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_COMPUTE_DEVICE_CODE)
    struct rename_function
    {
        HPX_NON_COPYABLE(rename_function);

        explicit rename_function(char const* name) {}
        template <typename F>
        explicit HPX_HOST_DEVICE rename_function(F && f) {}

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~rename_function() {}
    };
#elif HPX_HAVE_ITTNOTIFY != 0
    struct rename_function
    {
        HPX_NON_COPYABLE(rename_function);

        explicit rename_function(char const* name)
          : task_(thread_domain_,
                hpx::util::itt::string_handle(name))
        {}
        template <typename F>
        explicit rename_function(F && f)
          : task_(thread_domain_,
                hpx::traits::get_function_annotation_itt<
                    typename std::decay<F>::type
                >::call(f))
        {}

    private:
        hpx::util::itt::thread_domain thread_domain_;
        hpx::util::itt::task task_;
    };
#else
    struct rename_function
    {
        HPX_NON_COPYABLE(rename_function);

        explicit rename_function(char const* name)
          : desc_(hpx::threads::get_self_ptr() ?
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(), name) :
                nullptr)
        {
#if defined(HPX_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(
                external_timer::update_task(threads::get_self_timer_data(),
                std::string(name)));
#endif
        }

        template <typename F>
        explicit rename_function(F && f)
          : desc_(hpx::threads::get_self_ptr() ?
                hpx::threads::set_thread_description(
                    hpx::threads::get_self_id(),
                    hpx::util::thread_description(f)) :
                nullptr)
        {
#if defined(HPX_HAVE_APEX)
            /* no need to update the task description in APEX, because
             * this same description was used when the task was created. */
#endif
        }

        ~rename_function()
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

    namespace detail
    {
        template <typename F>
        struct annotated_function
        {
            annotated_function() noexcept
              : name_(nullptr)
            {}

            annotated_function(char const* name, F const& f)
              : name_(name), f_(f)
            {}

            annotated_function(char const* name, F && f)
              : name_(name), f_(std::move(f))
            {}

        public:
            template <typename ... Ts>
            typename invoke_result<
                typename util::decay_unwrap<F>::type, Ts...>::type
            operator()(Ts && ... ts)
            {
                rename_function(get_function_annotation());
                return util::invoke(f_, std::forward<Ts>(ts)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & f_;
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function address
            ///
            /// This function returns the passed function address.
            /// \param none

            std::size_t get_function_address() const
            {
                return traits::get_function_address<
                        typename util::decay_unwrap<F>::type
                    >::call(f_);
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function annotation
            ///
            /// This function returns the function annotation, if it has a name
            /// name is returned, name is returned; if name is empty the typeid
            /// is returned
            ///
            /// \param none
            char const* get_function_annotation() const noexcept
            {
                return name_ ? name_ : typeid(f_).name();
            }

        private:
            char const* name_;
            typename util::decay_unwrap<F>::type f_;
        };
    }

    template <typename F>
    detail::annotated_function<typename std::decay<F>::type>
    annotated_function(char const* name, F && f)
    {
        typedef detail::annotated_function<
            typename std::decay<F>::type
        > result_type;

        return result_type(name, std::forward<F>(f));
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    struct rename_function
    {
        HPX_NON_COPYABLE(rename_function);

        explicit rename_function(char const* /*name*/) {}
        template <typename F>
        explicit HPX_HOST_DEVICE rename_function(F && /*f*/) {}

        // add empty (but non-trivial) destructor to silence warnings
        HPX_HOST_DEVICE ~rename_function() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Given a function as an argument, the user can rename_function
    /// as well.
    /// Annotating includes setting the thread description per thread id.
    ///
    /// \param function
    template <typename F>
    F && annotated_function(char const*, F && f)
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
