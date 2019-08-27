/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang
    Copyright (c) 2007-2019 Hartmut Kaiser
    Copyright (c) Christopher Diggins 2005
    Copyright (c) Pablo Aguilar 2005
    Copyright (c) Kevlin Henney 2001

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    The class hpx::util::any is built based on boost::spirit::hold_any class.
    It adds support for HPX serialization, move assignment, == operator.
==============================================================================*/

#ifndef HPX_UTIL_SERIALIZABLE_ANY_HPP
#define HPX_UTIL_SERIALIZABLE_ANY_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/supports_streaming_with_any.hpp>

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <type_traits>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
#if defined(HPX_MSVC) && HPX_MSVC >= 1400
#pragma warning(push)
#pragma warning(disable : 4100)    // 'x': unreferenced formal parameter
#pragma warning(disable : 4127)    // conditional expression is constant
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail { namespace any {

    ////////////////////////////////////////////////////////////////////////////
    // serializable function pointer table
    template <typename IArch, typename OArch, typename Char>
    struct fxn_ptr_table
    {
        virtual ~fxn_ptr_table() {}
        virtual fxn_ptr_table* get_ptr() = 0;

        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void* const*, void**);
        void (*copy)(void* const*, void**);
        bool (*equal_to)(void* const*, void* const*);
        std::basic_istream<Char>& (*stream_in)(
            std::basic_istream<Char>&, void**);
        std::basic_ostream<Char>& (*stream_out)(
            std::basic_ostream<Char>&, void* const*);

        virtual void save_object(void* const*, OArch& ar, unsigned) = 0;
        virtual void load_object(void**, IArch& ar, unsigned) = 0;

        template <typename Arch>
        void serialize(Arch& ar, unsigned)
        {
        }

        HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(fxn_ptr_table);
    };

    ////////////////////////////////////////////////////////////////////////////
    template <typename IArch, typename OArch, typename Vtable, typename Char>
    struct fxn_ptr : fxn_ptr_table<IArch, OArch, Char>
    {
        using base_type = fxn_ptr_table<IArch, OArch, Char>;

        fxn_ptr()
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

        virtual base_type* get_ptr()
        {
            return Vtable::get_ptr();
        }

        void save_object(void* const* object, OArch& ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void** object, IArch& ar, unsigned)
        {
            ar & Vtable::construct(object);
        }

        template <typename Arch>
        void serialize(Arch& ar, unsigned)
        {
            ar & hpx::serialization::base_object<base_type>(*this);
        }
        HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(fxn_ptr);
    };
}}}}    // namespace hpx::util::detail::any

namespace hpx { namespace util {

    ////////////////////////////////////////////////////////////////////////////
    template <typename IArch, typename OArch, typename Char>
    class basic_any
    {
    public:
        // constructors
        basic_any() noexcept
          : table(
                detail::any::get_table<detail::any::empty>::template get<IArch,
                    OArch, Char>())
          , object(nullptr)
        {
        }

        basic_any(basic_any const& x)
          : table(
                detail::any::get_table<detail::any::empty>::template get<IArch,
                    OArch, Char>())
          , object(nullptr)
        {
            assign(x);
        }

        template <typename T>
        explicit basic_any(T const& x)
          : table(detail::any::get_table<typename util::decay<T>::type>::
                    template get<IArch, OArch, Char>())
          , object(nullptr)
        {
            using value_type = typename util::decay<T>::type;
            new_object(object, x,
                typename detail::any::get_table<value_type>::is_small());
        }

        // Move constructor
        basic_any(basic_any&& x) noexcept
          : table(x.table)
          , object(x.object)
        {
            x.table =
                detail::any::get_table<detail::any::empty>::template get<IArch,
                    OArch, Char>();
            x.object = nullptr;
        }

        // Perfect forwarding of T
        template <typename T>
        explicit basic_any(T&& x,
            typename std::enable_if<!std::is_same<basic_any,
                typename util::decay<T>::type>::value>::type* = nullptr)
          : table(detail::any::get_table<typename util::decay<T>::type>::
                    template get<IArch, OArch, Char>())
          , object(nullptr)
        {
            using value_type = typename util::decay<T>::type;
            new_object(object, std::forward<T>(x),
                typename detail::any::get_table<value_type>::is_small());
        }

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        // assignment
        basic_any& assign(basic_any const& x)
        {
            if (&x != this)
            {
                // are we copying between the same type?
                if (table == x.table)
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

        template <typename T>
        static void new_object(void*& object, T&& x, std::true_type)
        {
            using value_type = typename util::decay<T>::type;
            new (&object) value_type(std::forward<T>(x));
        }

        template <typename T>
        static void new_object(void*& object, T&& x, std::false_type)
        {
            using value_type = typename util::decay<T>::type;
            object = new value_type(std::forward<T>(x));
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
        template <typename T>
        basic_any& operator=(T&& rhs)
        {
            basic_any(std::forward<T>(rhs)).swap(*this);
            return *this;
        }

        // equality operator
        friend bool operator==(basic_any const& x, basic_any const& y)
        {
            if (&x == &y)    // same object
            {
                return true;
            }

            if (x.table == y.table)    // same type
            {
                return x.table->equal_to(
                    &x.object, &y.object);    // equal value?
            }

            return false;
        }

        template <typename T>
        friend bool operator==(basic_any const& b, T const& x)
        {
            using value_type = typename util::decay<T>::type;

            if (b.type() == typeid(value_type))    // same type
            {
                return b.cast<value_type>() == x;
            }

            return false;
        }

        // inequality operator
        friend bool operator!=(basic_any const& x, basic_any const& y)
        {
            return !(x == y);
        }

        template <typename T>
        friend bool operator!=(basic_any const& b, T const& x)
        {
            return !(b == x);
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
                throw bad_any_cast(type(), typeid(T));

            return detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

// implicit casting is disabled by default for compatibility with hpx::any
#ifdef HPX_ANY_IMPLICIT_CASTING
        // automatic casting operator
        template <typename T>
        operator T const&() const
        {
            return cast<T>();
        }
#endif    // implicit casting

        bool empty() const noexcept
        {
            return type() == typeid(detail::any::empty);
        }

        void reset()
        {
            if (!empty())
            {
                table->static_delete(&object);
                table = detail::any::get_table<
                    detail::any::empty>::template get<IArch, OArch, Char>();
                object = nullptr;
            }
        }

        // these functions have been added in the assumption that the embedded
        // type has a corresponding operator defined, which is completely safe
        // because hpx::util::any is used only in contexts where these operators
        // exist
        template <typename IArch_, typename OArch_, typename Char_>
        friend std::basic_istream<Char_>& operator>>(
            std::basic_istream<Char_>& i,
            basic_any<IArch_, OArch_, Char_>& obj);

        template <typename IArch_, typename OArch_, typename Char_>
        friend std::basic_ostream<Char_>& operator<<(
            std::basic_ostream<Char_>& o,
            basic_any<IArch_, OArch_, Char_> const& obj);

    private:
        friend class hpx::serialization::access;

        void load(IArch& ar, const unsigned version)
        {
            bool is_empty;
            ar & is_empty;

            if (is_empty)
            {
                reset();
            }
            else
            {
                typename detail::any::fxn_ptr_table<IArch, OArch, Char>* p =
                    nullptr;
                ar >> hpx::serialization::detail::raw_ptr(p);
                table = p->get_ptr();
                delete p;
                table->load_object(&object, ar, version);
            }
        }

        void save(OArch& ar, const unsigned version) const
        {
            bool is_empty = empty();
            ar & is_empty;
            if (!is_empty)
            {
                ar << hpx::serialization::detail::raw_ptr(table);
                table->save_object(&object, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER();

    private:    // types
        template <typename T, typename IArch_, typename OArch_,
            typename Char_>
        friend T* any_cast(basic_any<IArch_, OArch_, Char_>*) noexcept;

        // fields
        detail::any::fxn_ptr_table<IArch, OArch, Char>* table;
        void* object;
    };

    ////////////////////////////////////////////////////////////////////////////
    // backwards compatibility
    using any = basic_any<serialization::input_archive,
        serialization::output_archive, char>;
    using wany = basic_any<serialization::input_archive,
        serialization::output_archive, wchar_t>;

    ////////////////////////////////////////////////////////////////////////////
    // support for hashing any
    struct hash_any
    {
        template <typename Char>
        HPX_EXPORT std::size_t operator()(
            const basic_any<serialization::input_archive,
                serialization::output_archive, Char>& elem) const;
    };
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_MSVC) && HPX_MSVC >= 1400
#pragma warning(pop)
#endif

#endif
