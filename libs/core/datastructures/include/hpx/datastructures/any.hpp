/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang
    Copyright (c) 2007-2019 Hartmut Kaiser
    Copyright (c) Christopher Diggins 2005
    Copyright (c) Pablo Aguilar 2005
    Copyright (c) Kevlin Henney 2001

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    The class hpx::any is built based on boost::spirit::hold_any class.
    It adds support for HPX serialization, move assignment, == operator.
==============================================================================*/

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/traits/supports_streaming_with_any.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
#if defined(HPX_MSVC) && HPX_MSVC >= 1400
#pragma warning(push)
#pragma warning(disable : 4100)    // 'x': unreferenced formal parameter
#pragma warning(disable : 4127)    // conditional expression is constant
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ////////////////////////////////////////////////////////////////////////////
    struct bad_any_cast : std::bad_cast
    {
        bad_any_cast(std::type_info const& src, std::type_info const& dest)
          : from(src.name())
          , to(dest.name())
        {
        }

        const char* what() const noexcept override
        {
            return "bad any cast";
        }

        const char* from;
        const char* to;
    };
}    // namespace hpx

namespace hpx { namespace util {
    using bad_any_cast HPX_DEPRECATED_V(1, 6,
        "hpx::util::bad_any_cast is deprecated. Use hpx::bad_any_cast "
        "instead.") = hpx::bad_any_cast;
}}    // namespace hpx::util

namespace hpx { namespace util { namespace detail { namespace any {

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct get_table;

    // function pointer table
    template <typename IArch, typename OArch, typename Char, typename Copyable>
    struct fxn_ptr_table;

    template <>
    struct fxn_ptr_table<void, void, void, std::true_type>
    {
        virtual ~fxn_ptr_table() = default;
        virtual fxn_ptr_table* get_ptr() = 0;

        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void* const*, void**);
        void (*copy)(void* const*, void**);
        bool (*equal_to)(void* const*, void* const*);
    };

    template <>
    struct fxn_ptr_table<void, void, void, std::false_type>
    {
        virtual ~fxn_ptr_table() = default;
        virtual fxn_ptr_table* get_ptr() = 0;

        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        bool (*equal_to)(void* const*, void* const*);
    };

    ////////////////////////////////////////////////////////////////////////
    template <typename Char, typename Copyable>
    struct fxn_ptr_table<void, void, Char, Copyable>
      : fxn_ptr_table<void, void, void, Copyable>
    {
        virtual ~fxn_ptr_table() = default;
        fxn_ptr_table* get_ptr() override = 0;

        std::basic_istream<Char>& (*stream_in)(
            std::basic_istream<Char>&, void**);
        std::basic_ostream<Char>& (*stream_out)(
            std::basic_ostream<Char>&, void* const*);
    };

    ////////////////////////////////////////////////////////////////////////
    template <typename T, typename Small, typename Char,
        typename Enable = typename traits::supports_streaming_with_any<T>::type>
    struct streaming_base;

    // no streaming support
    template <typename T>
    struct streaming_base<T, std::true_type, void, std::true_type>
    {
    };

    template <typename T>
    struct streaming_base<T, std::true_type, void, std::false_type>
    {
    };
    template <typename T>
    struct streaming_base<T, std::false_type, void, std::true_type>
    {
    };

    template <typename T>
    struct streaming_base<T, std::false_type, void, std::false_type>
    {
    };

    // streaming support is enabled
    template <typename T, typename Char>
    struct streaming_base<T, std::true_type, Char, std::true_type>
    {
        template <typename Char_>
        static std::basic_istream<Char_>& stream_in(
            std::basic_istream<Char_>& i, void** obj)
        {
            i >> *reinterpret_cast<T*>(obj);
            return i;
        }

        template <typename Char_>
        static std::basic_ostream<Char_>& stream_out(
            std::basic_ostream<Char_>& o, void* const* obj)
        {
            o << *reinterpret_cast<T const*>(obj);
            return o;
        }
    };

    template <typename T, typename Char>
    struct streaming_base<T, std::false_type, Char, std::true_type>
    {
        template <typename Char_>
        static std::basic_istream<Char_>& stream_in(
            std::basic_istream<Char_>& i, void** obj)
        {
            i >> **reinterpret_cast<T**>(obj);
            return i;
        }

        template <typename Char_>
        static std::basic_ostream<Char_>& stream_out(
            std::basic_ostream<Char>& o, void* const* obj)
        {
            o << **reinterpret_cast<T* const*>(obj);
            return o;
        }
    };

    template <typename T, typename Small, typename Char>
    struct streaming_base<T, Small, Char, std::false_type>
    {
        template <typename Char_>
        static std::basic_istream<Char_>& stream_in(
            std::basic_istream<Char_>& i, void** /* obj */)
        {
            return i;
        }

        template <typename Char_>
        static std::basic_ostream<Char_>& stream_out(
            std::basic_ostream<Char_>& o, void* const* /* obj */)
        {
            return o;
        }
    };

    ////////////////////////////////////////////////////////////////////////
    // static functions for small value-types
    template <typename Small, typename Copyable>
    struct fxns;

    template <>
    struct fxns<std::true_type, std::true_type>
    {
        template <typename T, typename IArch, typename OArch, typename Char>
        struct type : public streaming_base<T, std::true_type, Char>
        {
            static fxn_ptr_table<IArch, OArch, Char, std::true_type>* get_ptr()
            {
                return detail::any::get_table<T>::template get<IArch, OArch,
                    Char, std::true_type>();
            }

            static std::type_info const& get_type()
            {
                return typeid(T);
            }
            static T& construct(void** f)
            {
                new (f) T;
                return *reinterpret_cast<T*>(f);
            }

            static T& get(void** f)
            {
                return *reinterpret_cast<T*>(f);
            }

            static T const& get(void* const* f)
            {
                return *reinterpret_cast<T const*>(f);
            }
            static void static_delete(void** x)
            {
                reinterpret_cast<T*>(x)->~T();
            }
            static void destruct(void** x)
            {
                reinterpret_cast<T*>(x)->~T();
            }
            static void clone(void* const* src, void** dest)
            {
                new (dest) T(*reinterpret_cast<T const*>(src));
            }
            static void copy(void* const* src, void** dest)
            {
                *reinterpret_cast<T*>(dest) = *reinterpret_cast<T const*>(src);
            }
            static bool equal_to(void* const* x, void* const* y)
            {
                return get(x) == get(y);
            }
        };
    };

    // static functions for big value-types (bigger than a void*)
    template <>
    struct fxns<std::false_type, std::true_type>
    {
        template <typename T, typename IArch, typename OArch, typename Char>
        struct type : public streaming_base<T, std::false_type, Char>
        {
            static fxn_ptr_table<IArch, OArch, Char, std::true_type>* get_ptr()
            {
                return detail::any::get_table<T>::template get<IArch, OArch,
                    Char, std::true_type>();
            }
            static std::type_info const& get_type()
            {
                return typeid(T);
            }
            static T& construct(void** f)
            {
                *f = new T;
                return **reinterpret_cast<T**>(f);
            }
            static T& get(void** f)
            {
                return **reinterpret_cast<T**>(f);
            }
            static T const& get(void* const* f)
            {
                return **reinterpret_cast<T* const*>(f);
            }
            static void static_delete(void** x)
            {
                // destruct and free memory
                delete (*reinterpret_cast<T**>(x));
            }
            static void destruct(void** x)
            {
                // destruct only, we'll reuse memory
                (*reinterpret_cast<T**>(x))->~T();
            }
            static void clone(void* const* src, void** dest)
            {
                *dest = new T(**reinterpret_cast<T* const*>(src));
            }
            static void copy(void* const* src, void** dest)
            {
                **reinterpret_cast<T**>(dest) =
                    **reinterpret_cast<T* const*>(src);
            }
            static bool equal_to(void* const* x, void* const* y)
            {
                return get(x) == get(y);
            }
        };
    };

    ////////////////////////////////////////////////////////////////////////
    // static functions for small value-types (unique_any)
    template <>
    struct fxns<std::true_type, std::false_type>
    {
        template <typename T, typename IArch, typename OArch, typename Char>
        struct type : public streaming_base<T, std::true_type, Char>
        {
            static fxn_ptr_table<IArch, OArch, Char, std::false_type>* get_ptr()
            {
                return detail::any::get_table<T>::template get<IArch, OArch,
                    Char, std::false_type>();
            }

            static std::type_info const& get_type()
            {
                return typeid(T);
            }
            static T& construct(void** f)
            {
                new (f) T;
                return *reinterpret_cast<T*>(f);
            }

            static T& get(void** f)
            {
                return *reinterpret_cast<T*>(f);
            }

            static T const& get(void* const* f)
            {
                return *reinterpret_cast<T const*>(f);
            }
            static void static_delete(void** x)
            {
                reinterpret_cast<T*>(x)->~T();
            }
            static void destruct(void** x)
            {
                reinterpret_cast<T*>(x)->~T();
            }
            static bool equal_to(void* const* x, void* const* y)
            {
                return get(x) == get(y);
            }
        };
    };

    // static functions for big value-types (bigger than a void*, unique)
    template <>
    struct fxns<std::false_type, std::false_type>
    {
        template <typename T, typename IArch, typename OArch, typename Char>
        struct type : public streaming_base<T, std::false_type, Char>
        {
            static fxn_ptr_table<IArch, OArch, Char, std::false_type>* get_ptr()
            {
                return detail::any::get_table<T>::template get<IArch, OArch,
                    Char, std::false_type>();
            }
            static std::type_info const& get_type()
            {
                return typeid(T);
            }
            static T& construct(void** f)
            {
                *f = new T;
                return **reinterpret_cast<T**>(f);
            }
            static T& get(void** f)
            {
                return **reinterpret_cast<T**>(f);
            }
            static T const& get(void* const* f)
            {
                return **reinterpret_cast<T* const*>(f);
            }
            static void static_delete(void** x)
            {
                // destruct and free memory
                delete (*reinterpret_cast<T**>(x));
            }
            static void destruct(void** x)
            {
                // destruct only, we'll reuse memory
                (*reinterpret_cast<T**>(x))->~T();
            }
            static bool equal_to(void* const* x, void* const* y)
            {
                return get(x) == get(y);
            }
        };
    };

    ////////////////////////////////////////////////////////////////////////
    template <typename IArch, typename OArch, typename Vtable, typename Char,
        typename Copyable>
    struct fxn_ptr;

    template <typename Vtable>
    struct fxn_ptr<void, void, Vtable, void, std::true_type>
      : fxn_ptr_table<void, void, void, std::true_type>
    {
        using base_type = fxn_ptr_table<void, void, void, std::true_type>;

        // this is constexpr starting C++14 only as older gcc's complain
        // about the constructor not having an empty body
        constexpr fxn_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::equal_to = Vtable::equal_to;
        }

        base_type* get_ptr() override
        {
            return Vtable::get_ptr();
        }
    };

    template <typename Vtable, typename Char>
    struct fxn_ptr<void, void, Vtable, Char, std::true_type>
      : fxn_ptr_table<void, void, Char, std::true_type>
    {
        using base_type = fxn_ptr_table<void, void, Char, std::true_type>;

        // this is constexpr starting C++14 only as older gcc's complain
        // about the constructor not having an empty body
        constexpr fxn_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::equal_to = Vtable::equal_to;
            base_type::stream_in = Vtable::stream_in;
            base_type::stream_out = Vtable::stream_out;
        }

        base_type* get_ptr() override
        {
            return Vtable::get_ptr();
        }
    };

    template <typename Vtable>
    struct fxn_ptr<void, void, Vtable, void, std::false_type>
      : fxn_ptr_table<void, void, void, std::false_type>
    {
        using base_type = fxn_ptr_table<void, void, void, std::false_type>;

        // this is constexpr starting C++14 only as older gcc's complain
        // about the constructor not having an empty body
        constexpr fxn_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::equal_to = Vtable::equal_to;
        }

        base_type* get_ptr() override
        {
            return Vtable::get_ptr();
        }
    };

    template <typename Vtable, typename Char>
    struct fxn_ptr<void, void, Vtable, Char, std::false_type>
      : fxn_ptr_table<void, void, Char, std::false_type>
    {
        using base_type = fxn_ptr_table<void, void, Char, std::false_type>;

        // this is constexpr starting C++14 only as older gcc's complain
        // about the constructor not having an empty body
        constexpr fxn_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::equal_to = Vtable::equal_to;
            base_type::stream_in = Vtable::stream_in;
            base_type::stream_out = Vtable::stream_out;
        }

        base_type* get_ptr() override
        {
            return Vtable::get_ptr();
        }
    };

    ////////////////////////////////////////////////////////////////////////
    template <typename Vtable, typename T>
    struct any_vtable
    {
        static_assert(
            !std::is_reference<T>::value, "T shall have no ref-qualifiers");

        static Vtable* call()
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
            return nullptr;
#else
            static Vtable instance{};
            return &instance;
#endif
        }
    };

    template <typename T>
    struct get_table
    {
        using is_small =
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            std::integral_constant<bool, (sizeof(T) <= sizeof(void*))>;

        template <typename IArch, typename OArch, typename Char,
            typename Copyable>
        static constexpr fxn_ptr_table<IArch, OArch, Char, Copyable>* get()
        {
            using fxn_type = typename fxns<is_small, Copyable>::template type<T,
                IArch, OArch, Char>;

            using fxn_ptr_type =
                fxn_ptr<IArch, OArch, fxn_type, Char, Copyable>;
            return any_vtable<fxn_ptr_type, T>::call();
        }
    };

    ////////////////////////////////////////////////////////////////////////
    struct empty
    {
        bool operator==(empty const&) const
        {
            return false;    // undefined
        }
        bool operator!=(empty const&) const
        {
            return false;    // undefined
        }
    };

    template <typename Char>
    inline std::basic_istream<Char>& operator>>(
        std::basic_istream<Char>& i, empty&)
    {
        // If this assertion fires you tried to insert from a std istream
        // into an empty any instance. This simply can't work, because
        // there is no way to figure out what type to extract from the
        // stream.
        // The only way to make this work is to assign an arbitrary
        // value of the required type to the any instance you want to
        // stream to. This assignment has to be executed before the actual
        // call to the operator>>().
        HPX_ASSERT(false &&
            "Tried to insert from a std istream into an empty "
            "any instance");
        return i;
    }

    template <typename Char>
    inline std::basic_ostream<Char>& operator<<(
        std::basic_ostream<Char>& o, empty const&)
    {
        return o;
    }

    // helper types allowing to access internal data of basic_any
    struct stream_support;
    struct any_cast_support;

}}}}    // namespace hpx::util::detail::any

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename IArch, typename OArch, typename Char = char,
        typename Copyable = std::true_type>
    class basic_any;

    ////////////////////////////////////////////////////////////////////////////
    // specialization for copyable any without streaming and without
    // serialization
    template <>
    class basic_any<void, void, void, std::true_type>
    {
    public:
        // constructors
        constexpr basic_any() noexcept
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, void, std::true_type>())
          , object(nullptr)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, void, std::true_type>())
          , object(nullptr)
        {
            assign(x);
        }

        // Move constructor
        basic_any(basic_any&& x) noexcept
          : table(x.table)
          , object(x.object)
        {
            x.object = nullptr;
            x.table =
                detail::any::get_table<detail::any::empty>::template get<void,
                    void, void, std::true_type>();
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<!std::is_same<basic_any,
                typename std::decay<T>::type>::value>::type>
        basic_any(T&& x,
            typename std::enable_if<std::is_copy_constructible<
                typename std::decay<T>::type>::value>::type* = nullptr)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<T>(x));
        }

#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
        template <typename T, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(std::in_place_type_t<T>, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<Ts>(ts)...);
        }

        template <typename T, typename U, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(
            std::in_place_type_t<T>, std::initializer_list<U> il, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(), il,
                std::forward<Ts>(ts)...);
        }
#endif

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        basic_any& assign(basic_any const& x)
        {
            if (&x != this)
            {
                // are we copying between the same type?
                if (type() == x.type())
                {
                    // if so, we can avoid reallocation
                    table->copy(&x.object, &object);
                }
                else
                {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::true_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            new (&object) value_type(std::forward<Ts>(ts)...);
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::false_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            object = new value_type(std::forward<Ts>(ts)...);
        }

    public:
        // copy assignment operator
        basic_any& operator=(basic_any const& x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        // move assignment
        basic_any& operator=(basic_any&& rhs) noexcept
        {
            rhs.swap(*this);
            basic_any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<
                !std::is_same<basic_any, typename std::decay<T>::type>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        basic_any& operator=(T&& rhs)
        {
            basic_any(std::forward<T>(rhs)).swap(*this);
            return *this;
        }

        // utility functions
        basic_any& swap(basic_any& x) noexcept
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        std::type_info const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != typeid(T))
                throw hpx::bad_any_cast(type(), typeid(T));

            return hpx::util::detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

        bool has_value() const noexcept
        {
            return type() != typeid(detail::any::empty);
        }

        void reset()
        {
            if (has_value())
            {
                table->static_delete(&object);
                table =
                    detail::any::get_table<detail::any::empty>::template get<
                        void, void, void, std::true_type>();
                object = nullptr;
            }
        }

        // equality operator
        bool equal_to(basic_any const& rhs) const noexcept
        {
            if (this == &rhs)    // same object
            {
                return true;
            }

            if (type() == rhs.type())    // same type
            {
                return table->equal_to(&object, &rhs.object);    // equal value?
            }

            return false;
        }

    private:    // types
        friend struct detail::any::any_cast_support;

        // fields
        detail::any::fxn_ptr_table<void, void, void, std::true_type>* table;
        void* object;
    };

    ////////////////////////////////////////////////////////////////////////////
    // specialization for hpx::any supporting streaming
    template <typename Char>    // default is char
    class basic_any<void, void, Char, std::true_type>
    {
    public:
        // constructors
        constexpr basic_any() noexcept
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, Char, std::true_type>())
          , object(nullptr)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, Char, std::true_type>())
          , object(nullptr)
        {
            assign(x);
        }

        // Move constructor
        basic_any(basic_any&& x) noexcept
          : table(x.table)
          , object(x.object)
        {
            x.object = nullptr;
            x.table =
                detail::any::get_table<detail::any::empty>::template get<void,
                    void, Char, std::true_type>();
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<!std::is_same<basic_any,
                typename std::decay<T>::type>::value>::type>
        basic_any(T&& x,
            typename std::enable_if<std::is_copy_constructible<
                typename std::decay<T>::type>::value>::type* = nullptr)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<T>(x));
        }

#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
        template <typename T, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(std::in_place_type_t<T>, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<Ts>(ts)...);
        }

        template <typename T, typename U, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(
            std::in_place_type_t<T>, std::initializer_list<U> il, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::true_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(), il,
                std::forward<Ts>(ts)...);
        }
#endif

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        basic_any& assign(basic_any const& x)
        {
            if (&x != this)
            {
                // are we copying between the same type?
                if (type() == x.type())
                {
                    // if so, we can avoid reallocation
                    table->copy(&x.object, &object);
                }
                else
                {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::true_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            new (&object) value_type(std::forward<Ts>(ts)...);
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::false_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            object = new value_type(std::forward<Ts>(ts)...);
        }

    public:
        // copy assignment operator
        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        basic_any& operator=(basic_any const& x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        // move assignment
        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        basic_any& operator=(basic_any&& rhs) noexcept
        {
            rhs.swap(*this);
            basic_any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<
                !std::is_same<basic_any, typename std::decay<T>::type>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        basic_any& operator=(T&& rhs) noexcept
        {
            basic_any(std::forward<T>(rhs)).swap(*this);
            return *this;
        }

        // utility functions
        basic_any& swap(basic_any& x) noexcept
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        std::type_info const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != typeid(T))
                throw hpx::bad_any_cast(type(), typeid(T));

            return hpx::util::detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

        bool has_value() const noexcept
        {
            return type() != typeid(detail::any::empty);
        }

        void reset()
        {
            if (has_value())
            {
                table->static_delete(&object);
                table =
                    detail::any::get_table<detail::any::empty>::template get<
                        void, void, Char, std::true_type>();
                object = nullptr;
            }
        }

        // equality operator
        bool equal_to(basic_any const& rhs) const noexcept
        {
            if (this == &rhs)    // same object
            {
                return true;
            }

            if (type() == rhs.type())    // same type
            {
                return table->equal_to(&object, &rhs.object);    // equal value?
            }

            return false;
        }

    private:    // types
        friend struct detail::any::any_cast_support;
        friend struct detail::any::stream_support;

        // fields
        detail::any::fxn_ptr_table<void, void, Char, std::true_type>* table;
        void* object;
    };

    ////////////////////////////////////////////////////////////////////////////
    // specialization for unique_any without streaming and without
    // serialization
    template <>
    class basic_any<void, void, void, std::false_type>
    {
    public:
        // constructors
        constexpr basic_any() noexcept
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, void, std::false_type>())
          , object(nullptr)
        {
        }

        // Move constructor
        basic_any(basic_any&& x) noexcept
          : table(x.table)
          , object(x.object)
        {
            x.object = nullptr;
            x.table =
                detail::any::get_table<detail::any::empty>::template get<void,
                    void, void, std::false_type>();
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<!std::is_same<basic_any,
                typename std::decay<T>::type>::value>::type>
        basic_any(T&& x,
            typename std::enable_if<std::is_move_constructible<
                typename std::decay<T>::type>::value>::type* = nullptr)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<T>(x));
        }

#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
        template <typename T, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(std::in_place_type_t<T>, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<Ts>(ts)...);
        }

        template <typename T, typename U, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(
            std::in_place_type_t<T>, std::initializer_list<U> il, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, void, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(), il,
                std::forward<Ts>(ts)...);
        }
#endif

        basic_any(basic_any const& x) = delete;
        basic_any& operator=(basic_any const& x) = delete;

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        template <typename T, typename... Ts>
        static void new_object(void*& object, std::true_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            new (&object) value_type(std::forward<Ts>(ts)...);
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::false_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            object = new value_type(std::forward<Ts>(ts)...);
        }

    public:
        // move assignment
        basic_any& operator=(basic_any&& rhs) noexcept
        {
            rhs.swap(*this);
            basic_any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<
                !std::is_same<basic_any, typename std::decay<T>::type>::value &&
                std::is_move_constructible<
                    typename std::decay<T>::type>::value>::type>
        basic_any& operator=(T&& rhs)
        {
            basic_any(std::forward<T>(rhs)).swap(*this);
            return *this;
        }

        // utility functions
        basic_any& swap(basic_any& x) noexcept
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        std::type_info const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != typeid(T))
                throw hpx::bad_any_cast(type(), typeid(T));

            return hpx::util::detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

        bool has_value() const noexcept
        {
            return type() != typeid(detail::any::empty);
        }

        void reset()
        {
            if (has_value())
            {
                table->static_delete(&object);
                table =
                    detail::any::get_table<detail::any::empty>::template get<
                        void, void, void, std::false_type>();
                object = nullptr;
            }
        }

        // equality operator
        bool equal_to(basic_any const& rhs) const noexcept
        {
            if (this == &rhs)    // same object
            {
                return true;
            }

            if (type() == rhs.type())    // same type
            {
                return table->equal_to(&object, &rhs.object);    // equal value?
            }

            return false;
        }

    private:    // types
        friend struct detail::any::any_cast_support;

        // fields
        detail::any::fxn_ptr_table<void, void, void, std::false_type>* table;
        void* object;
    };

    // specialization for unique_any supporting streaming
    template <typename Char>    // default is char
    class basic_any<void, void, Char, std::false_type>
    {
    public:
        // constructors
        constexpr basic_any() noexcept
          : table(detail::any::get_table<detail::any::empty>::template get<void,
                void, Char, std::false_type>())
          , object(nullptr)
        {
        }

        // Move constructor
        basic_any(basic_any&& x) noexcept
          : table(x.table)
          , object(x.object)
        {
            x.object = nullptr;
            x.table =
                detail::any::get_table<detail::any::empty>::template get<void,
                    void, Char, std::false_type>();
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<!std::is_same<basic_any,
                typename std::decay<T>::type>::value>::type>
        basic_any(T&& x,
            typename std::enable_if<std::is_move_constructible<
                typename std::decay<T>::type>::value>::type* = nullptr)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<T>(x));
        }

#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
        template <typename T, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(std::in_place_type_t<T>, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(),
                std::forward<Ts>(ts)...);
        }

        template <typename T, typename U, typename... Ts,
            typename Enable = typename std::enable_if<
                std::is_constructible<typename std::decay<T>::type,
                    Ts...>::value &&
                std::is_copy_constructible<
                    typename std::decay<T>::type>::value>::type>
        explicit basic_any(
            std::in_place_type_t<T>, std::initializer_list<U> il, Ts&&... ts)
          : table(detail::any::get_table<typename std::decay<T>::type>::
                    template get<void, void, Char, std::false_type>())
          , object(nullptr)
        {
            using value_type = typename std::decay<T>::type;
            new_object<T>(object,
                typename detail::any::get_table<value_type>::is_small(), il,
                std::forward<Ts>(ts)...);
        }
#endif

        basic_any(basic_any const& x) = delete;
        basic_any& operator=(basic_any const& x) = delete;

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        template <typename T, typename... Ts>
        static void new_object(void*& object, std::true_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            new (&object) value_type(std::forward<Ts>(ts)...);
        }

        template <typename T, typename... Ts>
        static void new_object(void*& object, std::false_type, Ts&&... ts)
        {
            using value_type = typename std::decay<T>::type;
            object = new value_type(std::forward<Ts>(ts)...);
        }

    public:
        // move assignment
        basic_any& operator=(basic_any&& rhs) noexcept
        {
            rhs.swap(*this);
            basic_any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of T
        template <typename T,
            typename Enable = typename std::enable_if<
                !std::is_same<basic_any, typename std::decay<T>::type>::value &&
                std::is_move_constructible<
                    typename std::decay<T>::type>::value>::type>
        basic_any& operator=(T&& rhs) noexcept
        {
            basic_any(std::forward<T>(rhs)).swap(*this);
            return *this;
        }

        // utility functions
        basic_any& swap(basic_any& x) noexcept
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        std::type_info const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != typeid(T))
                throw hpx::bad_any_cast(type(), typeid(T));

            return hpx::util::detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

        bool has_value() const noexcept
        {
            return type() != typeid(detail::any::empty);
        }

        void reset()
        {
            if (has_value())
            {
                table->static_delete(&object);
                table =
                    detail::any::get_table<detail::any::empty>::template get<
                        void, void, Char, std::false_type>();
                object = nullptr;
            }
        }

        // equality operator
        bool equal_to(basic_any const& rhs) const noexcept
        {
            if (this == &rhs)    // same object
            {
                return true;
            }

            if (type() == rhs.type())    // same type
            {
                return table->equal_to(&object, &rhs.object);    // equal value?
            }

            return false;
        }

    private:    // types
        friend struct detail::any::any_cast_support;
        friend struct detail::any::stream_support;

        // fields
        detail::any::fxn_ptr_table<void, void, Char, std::false_type>* table;
        void* object;
    };

    ////////////////////////////////////////////////////////////////////////////
    namespace detail { namespace any {

        struct any_cast_support
        {
            template <typename T, typename IArch, typename OArch, typename Char,
                typename Copyable>
            static T* call(
                basic_any<IArch, OArch, Char, Copyable>* operand) noexcept
            {
                return get_table<T>::is_small::value ?
                    reinterpret_cast<T*>(
                        reinterpret_cast<void*>(&operand->object)) :
                    reinterpret_cast<T*>(
                        reinterpret_cast<void*>(operand->object));
            }
        };

        struct stream_support
        {
            template <typename IArch, typename OArch, typename Char,
                typename Copyable>
            static std::basic_istream<Char>& stream_in(
                std::basic_istream<Char>& i,
                basic_any<IArch, OArch, Char, Copyable>& obj)
            {
                return obj.table->stream_in(i, &obj.object);
            }

            template <typename IArch, typename OArch, typename Char,
                typename Copyable>
            static std::basic_ostream<Char>& stream_out(
                std::basic_ostream<Char>& o,
                basic_any<IArch, OArch, Char, Copyable> const& obj)
            {
                return obj.table->stream_out(o, &obj.object);
            }
        };
    }}    // namespace detail::any

    template <typename IArch, typename OArch, typename Char, typename Copyable,
        typename Enable =
            typename std::enable_if<!std::is_void<Char>::value>::type>
    std::basic_istream<Char>& operator>>(std::basic_istream<Char>& i,
        basic_any<IArch, OArch, Char, Copyable>& obj)
    {
        return detail::any::stream_support::stream_in(i, obj);
    }

    template <typename IArch, typename OArch, typename Char, typename Copyable,
        typename Enable =
            typename std::enable_if<!std::is_void<Char>::value>::type>
    std::basic_ostream<Char>& operator<<(std::basic_ostream<Char>& o,
        basic_any<IArch, OArch, Char, Copyable> const& obj)
    {
        return detail::any::stream_support::stream_out(o, obj);
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename IArch, typename OArch, typename Char, typename Copyable>
    void swap(basic_any<IArch, OArch, Char, Copyable>& lhs,
        basic_any<IArch, OArch, Char, Copyable>& rhs) noexcept
    {
        lhs.swap(rhs);
    }
}}    // namespace hpx::util

namespace hpx {
#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
    template <typename T, typename... Ts>
    util::basic_any<void, void, void, std::true_type> make_any_nonser(
        Ts&&... ts)
    {
        return util::basic_any<void, void, void, std::true_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename U, typename... Ts>
    util::basic_any<void, void, void, std::true_type> make_any_nonser(
        std::initializer_list<U> il, Ts&&... ts)
    {
        return util::basic_any<void, void, void, std::true_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }

    template <typename T, typename... Ts>
    util::basic_any<void, void, void, std::false_type> make_unique_any_nonser(
        Ts&&... ts)
    {
        return util::basic_any<void, void, void, std::false_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename U, typename... Ts>
    util::basic_any<void, void, void, std::false_type> make_unique_any_nonser(
        std::initializer_list<U> il, Ts&&... ts)
    {
        return util::basic_any<void, void, void, std::false_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }
#endif

    template <typename T>
    util::basic_any<void, void, void, std::true_type> make_any_nonser(T&& t)
    {
        return util::basic_any<void, void, void, std::true_type>(
            std::forward<T>(t));
    }

    template <typename T>
    util::basic_any<void, void, void, std::false_type> make_unique_any_nonser(
        T&& t)
    {
        return util::basic_any<void, void, void, std::false_type>(
            std::forward<T>(t));
    }

    using any_nonser = util::basic_any<void, void, void, std::true_type>;
    using unique_any_nonser =
        util::basic_any<void, void, void, std::false_type>;

    template <typename T, typename IArch, typename OArch, typename Char,
        typename Copyable>
    inline T* any_cast(
        util::basic_any<IArch, OArch, Char, Copyable>* operand) noexcept
    {
        if (operand && operand->type() == typeid(T))
        {
            return util::detail::any::any_cast_support::template call<T>(
                operand);
        }
        return nullptr;
    }

    template <typename T, typename IArch, typename OArch, typename Char,
        typename Copyable>
    inline T const* any_cast(
        util::basic_any<IArch, OArch, Char, Copyable> const* operand) noexcept
    {
        return hpx::any_cast<T>(
            const_cast<util::basic_any<IArch, OArch, Char, Copyable>*>(
                operand));
    }

    template <typename T, typename IArch, typename OArch, typename Char,
        typename Copyable>
    T any_cast(util::basic_any<IArch, OArch, Char, Copyable>& operand)
    {
        using nonref = typename std::remove_reference<T>::type;

        nonref* result = hpx::any_cast<nonref>(&operand);
        if (!result)
            throw hpx::bad_any_cast(operand.type(), typeid(T));
        return static_cast<T>(*result);
    }

    template <typename T, typename IArch, typename OArch, typename Char,
        typename Copyable>
    T const& any_cast(
        util::basic_any<IArch, OArch, Char, Copyable> const& operand)
    {
        using nonref = typename std::remove_reference<T>::type;

        return hpx::any_cast<nonref const&>(
            const_cast<util::basic_any<IArch, OArch, Char, Copyable>&>(
                operand));
    }
}    // namespace hpx

namespace hpx { namespace util {
#if defined(HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T)
    ////////////////////////////////////////////////////////////////////////////
    // make copyable any
    template <typename T, typename... Ts>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_any_nonser is deprecated. Please use "
        "hpx::make_any_nonser instead.")
    basic_any<void, void, void, std::true_type> make_any_nonser(Ts&&... ts)
    {
        return basic_any<void, void, void, std::true_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename U, typename... Ts>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_any_nonser is deprecated. Please use "
        "hpx::make_any_nonser instead.")
    basic_any<void, void, void, std::true_type> make_any_nonser(
        std::initializer_list<U> il, Ts&&... ts)
    {
        return basic_any<void, void, void, std::true_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }

    template <typename T, typename Char, typename... Ts>
    basic_any<void, void, Char, std::true_type> make_streamable_any_nonser(
        Ts&&... ts)
    {
        return basic_any<void, void, Char, std::true_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename Char, typename U, typename... Ts>
    basic_any<void, void, Char, std::true_type> make_streamable_any_nonser(
        std::initializer_list<U> il, Ts&&... ts)
    {
        return basic_any<void, void, Char, std::true_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }

    ////////////////////////////////////////////////////////////////////////////
    // make unique_any
    template <typename T, typename... Ts>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_unique_any_nonser is deprecated. Please use "
        "hpx::make_unique_any_nonser instead.")
    basic_any<void, void, void, std::false_type> make_unique_any_nonser(
        Ts&&... ts)
    {
        return basic_any<void, void, void, std::false_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename U, typename... Ts>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_unique_any_nonser is deprecated. Please use "
        "hpx::make_unique_any_nonser instead.")
    basic_any<void, void, void, std::false_type> make_unique_any_nonser(
        std::initializer_list<U> il, Ts&&... ts)
    {
        return basic_any<void, void, void, std::false_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }

    template <typename T, typename Char, typename... Ts>
    basic_any<void, void, Char, std::false_type>
    make_streamable_unique_any_nonser(Ts&&... ts)
    {
        return basic_any<void, void, Char, std::false_type>(
            std::in_place_type<T>, std::forward<Ts>(ts)...);
    }

    template <typename T, typename Char, typename U, typename... Ts>
    basic_any<void, void, Char, std::false_type>
    make_streamable_unique_any_nonser(std::initializer_list<U> il, Ts&&... ts)
    {
        return basic_any<void, void, Char, std::false_type>(
            std::in_place_type<T>, il, std::forward<Ts>(ts)...);
    }
#endif

    // make copyable any
    template <typename T>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_any_nonser is deprecated. Please use "
        "hpx::make_any_nonser instead.")
    basic_any<void, void, void, std::true_type> make_any_nonser(T&& t)
    {
        return basic_any<void, void, void, std::true_type>(std::forward<T>(t));
    }

    template <typename T, typename Char>
    basic_any<void, void, Char, std::true_type> make_streamable_any_nonser(
        T&& t)
    {
        return basic_any<void, void, Char, std::true_type>(std::forward<T>(t));
    }

    // make unique_any
    template <typename T>
    HPX_DEPRECATED_V(1, 6,
        "hpx::util::make_unique_any_nonser is deprecated. Please use "
        "hpx::make_unique_any_nonser instead.")
    basic_any<void, void, void, std::false_type> make_unique_any_nonser(T&& t)
    {
        return basic_any<void, void, void, std::false_type>(std::forward<T>(t));
    }

    template <typename T, typename Char>
    basic_any<void, void, Char, std::false_type>
    make_streamable_unique_any_nonser(T&& t)
    {
        return basic_any<void, void, Char, std::false_type>(std::forward<T>(t));
    }

    ////////////////////////////////////////////////////////////////////////////
    // better names for copyable any
    using any_nonser HPX_DEPRECATED_V(1, 6,
        "hpx::util::any_nonser is deprecated. Please use hpx::any_nonser "
        "instead.") = basic_any<void, void, void, std::true_type>;

    using streamable_any_nonser = basic_any<void, void, char, std::true_type>;
    using streamable_wany_nonser =
        basic_any<void, void, wchar_t, std::true_type>;

    ////////////////////////////////////////////////////////////////////////////
    // better names for unique_any
    using unique_any_nonser HPX_DEPRECATED_V(1, 6,
        "hpx::util::unique_any_nonser is deprecated. Please use "
        "hpx::unique_any_nonser instead.") =
        basic_any<void, void, void, std::false_type>;

    using streamable_unique_any_nonser =
        basic_any<void, void, char, std::false_type>;
    using streamable_unique_wany_nonser =
        basic_any<void, void, wchar_t, std::false_type>;

}}    // namespace hpx::util

////////////////////////////////////////////////////////////////////////////////
#if defined(HPX_MSVC) && HPX_MSVC >= 1400
#pragma warning(pop)
#endif
