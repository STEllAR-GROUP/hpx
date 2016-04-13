//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_DESCRIPTION_FEB_19_2016_0200PM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_DESCRIPTION_FEB_19_2016_0200PM

#include <hpx/config.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/assert.hpp>
#ifndef HPX_HAVE_CXX11_EXPLICIT_CONVERSION_OPERATORS
#include <hpx/util/safe_bool.hpp>
#endif

#include <iosfwd>
#include <string>
#include <utility>
#include <type_traits>

namespace hpx { namespace util
{
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    struct thread_description
    {
    public:
        enum data_type
        {
            data_type_description = 0,
            data_type_address = 1
        };

    private:
        union data
        {
            char const* desc_;
            std::size_t addr_;
        };

        data_type type_;
        data data_;

#if !defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        HPX_EXPORT void init_from_alternative_name(char const* altname);
#endif

    public:
        thread_description() HPX_NOEXCEPT
          : type_(data_type_description)
        {
            data_.desc_ = 0;
        }

        thread_description(char const* desc) HPX_NOEXCEPT
          : type_(data_type_description)
        {
            data_.desc_ = desc;
        }

        template <typename F, typename =
            typename std::enable_if<
                !std::is_same<F, thread_description>::value &&
                !traits::is_action<F>::value
            >::type>
        explicit thread_description(F const& f, char const* altname = 0) HPX_NOEXCEPT
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
            type_ = data_type_address;
            data_.addr_ = traits::get_function_address<F>::call(f);
#else
            init_from_alternative_name(altname);
#endif
        }

        template <typename Action, typename =
            typename std::enable_if<
                traits::is_action<Action>::value
            >::type>
        explicit thread_description(Action,
                char const* altname = 0) HPX_NOEXCEPT
          : type_(data_type_description)
        {
            data_.desc_ = hpx::actions::detail::get_action_name<Action>();
        }

        data_type kind() const HPX_NOEXCEPT
        {
            return type_;
        }

        char const* get_description() const HPX_NOEXCEPT
        {
            HPX_ASSERT(type_ == data_type_description);
            return data_.desc_;
        }

        std::size_t get_address() const HPX_NOEXCEPT
        {
            HPX_ASSERT(type_ == data_type_address);
            return data_.addr_;
        }

#ifdef HPX_HAVE_CXX11_EXPLICIT_CONVERSION_OPERATORS
        explicit operator bool() const HPX_NOEXCEPT
        {
            return valid();
        }
#else
        operator typename util::safe_bool<thread_description>
            ::result_type() const HPX_NOEXCEPT
        {
            return util::safe_bool<thread_description>()(valid());
        }
#endif

        bool valid() const HPX_NOEXCEPT
        {
            if (type_ == data_type_description)
                return 0 != data_.desc_;

            HPX_ASSERT(type_ == data_type_address);
            return 0 != data_.addr_;
        }
    };
#else
    ///////////////////////////////////////////////////////////////////////////
    struct thread_description
    {
    public:
        enum data_type
        {
            data_type_description = 0,
            data_type_address = 1
        };

    public:
        thread_description() HPX_NOEXCEPT
        {
        }

        thread_description(char const* desc) HPX_NOEXCEPT
        {
        }

        template <typename F, typename =
            typename std::enable_if<
                !std::is_same<F, thread_description>::value &&
                !traits::is_action<F>::value
            >::type>
        explicit thread_description(F const& f,
            char const* altname = 0) HPX_NOEXCEPT
        {
        }

        template <typename Action, typename =
            typename std::enable_if<
                traits::is_action<Action>::value
            >::type>
        explicit thread_description(Action,
            char const* altname = 0) HPX_NOEXCEPT
        {
        }

        data_type kind() const HPX_NOEXCEPT
        {
            return data_type_description;
        }

        char const* get_description() const HPX_NOEXCEPT
        {
            return "<unknown>";
        }

        std::size_t get_address() const HPX_NOEXCEPT
        {
            return 0;
        }

#ifdef HPX_HAVE_CXX11_EXPLICIT_CONVERSION_OPERATORS
        explicit operator bool() const HPX_NOEXCEPT
        {
            return valid();
        }
#else
        operator typename util::safe_bool<thread_description>
            ::result_type() const HPX_NOEXCEPT
        {
            return util::safe_bool<thread_description>()(valid());
        }
#endif

        bool valid() const HPX_NOEXCEPT
        {
            return true;
        }
    };
#endif

    HPX_EXPORT std::ostream& operator<<(std::ostream&, thread_description const&);
    HPX_EXPORT std::string as_string(thread_description const& desc);
}}

#endif

