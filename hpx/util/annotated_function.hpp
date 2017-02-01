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
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/thread_description.hpp>
#endif

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    namespace detail
    {
        template <typename F>
        struct annotated_function
        {
            annotated_function()
              : name_(nullptr)
            {}

            annotated_function(F const& f, char const* name)
              : f_(f), name_(name)
            {}

            annotated_function(F && f, char const* name)
              : f_(std::move(f)), name_(name)
            {}

        private:
            struct reset_name
            {
                reset_name(annotated_function& f)
                  : f_(f), desc_()
                {
                    if (f_.name_)
                    {
                        desc_ = threads::set_thread_description(
                            threads::get_self_id(), f_.name_);
                    }
                }
                ~reset_name()
                {
                    if (desc_)
                    {
                        threads::set_thread_description(
                            threads::get_self_id(), desc_);
                    }
                }

                annotated_function& f_;
                util::thread_description desc_;
            };
            friend struct reset_name;

        public:
            template <typename ... Ts>
            typename deferred_result_of<F(Ts...)>::type
            operator()(Ts && ... ts)
            {
                reset_name on_exit(*this);
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

            char const* get_function_annotation() const
            {
                return name_ ? name_ : "<unknown>";
            }

        private:
            typename util::decay_unwrap<F>::type f_;
            char const* name_;
        };
    }

    template <typename F>
    detail::annotated_function<F>
    annotated_function(F && f, char const* name = nullptr)
    {
        return detail::annotated_function<F>(std::forward<F>(f), name);
    }
#else
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
        static std::size_t call(util::detail::annotated_function<F> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<util::detail::annotated_function<F> >
    {
        static char const* call(util::detail::annotated_function<F> const& f) HPX_NOEXCEPT
        {
            return f.get_function_annotation();
        }
    };
}}
#endif

#endif
