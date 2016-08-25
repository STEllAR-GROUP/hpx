/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang
    Copyright (c) 2007-2013 Hartmut Kaiser
    Copyright (c) Christopher Diggins 2005
    Copyright (c) Pablo Aguilar 2005
    Copyright (c) Kevlin Henney 2001

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    The class hpx::util::any is built based on boost::spirit::hold_any class.
    It adds support for HPX serialization, move assignment, == operator.
==============================================================================*/
#ifndef HPX_UTIL_ANY_HPP
#define HPX_UTIL_ANY_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/supports_streaming_with_any.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>

#include <boost/detail/sp_typeinfo.hpp>
#include <boost/functional/hash.hpp>
#include <boost/throw_exception.hpp>

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#if BOOST_WORKAROUND(HPX_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4100)   // 'x': unreferenced formal parameter
# pragma warning(disable: 4127)   // conditional expression is constant
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    struct bad_any_cast
      : std::bad_cast
    {
        bad_any_cast(boost::detail::sp_typeinfo const& src,
                boost::detail::sp_typeinfo const& dest)
          : from(src.name()), to(dest.name())
        {}

        virtual const char* what() const throw() { return "bad any cast"; }

        const char* from;
        const char* to;
    };

    namespace detail { namespace any
    {
        template <typename T>
        struct get_table;

        // serializable function pointer table
        template <typename IArchive, typename OArchive, typename Char>
        struct fxn_ptr_table
        {
            virtual ~fxn_ptr_table() {}
            virtual fxn_ptr_table * get_ptr() = 0;

            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*copy)(void* const*, void**);
            bool (*equal_to)(void* const*, void* const*);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&,
                void* const*);

            virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
            virtual void load_object(void **, IArchive & ar, unsigned) = 0;

            template <typename Archive>
            void serialize(Archive & ar, unsigned) {}

            HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(fxn_ptr_table);
        };

        // function pointer table
        template <typename Char>
        struct fxn_ptr_table<void, void, Char>
        {
            virtual ~fxn_ptr_table() {}
            virtual fxn_ptr_table * get_ptr() = 0;

            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*copy)(void* const*, void**);
            bool (*equal_to)(void* const*, void* const*);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&,
                void* const*);
        };

        template <typename T
          , typename Small
          , typename Enable = typename traits::supports_streaming_with_any<T>::type>
        struct streaming_base;

        template <typename T>
        struct streaming_base<T, std::true_type, std::true_type>
        {
            template <typename Char>
            static std::basic_istream<Char>&
            stream_in (std::basic_istream<Char>& i, void** obj)
            {
                i >> *reinterpret_cast<T*>(obj);
                return i;
            }

            template <typename Char>
            static std::basic_ostream<Char>&
            stream_out(std::basic_ostream<Char>& o, void* const* obj)
            {
                o << *reinterpret_cast<T const*>(obj);
                return o;
            }
        };

        template <typename T>
        struct streaming_base<T, std::false_type, std::true_type>
        {
            template <typename Char>
            static std::basic_istream<Char>&
            stream_in (std::basic_istream<Char>& i, void** obj)
            {
                i >> **reinterpret_cast<T**>(obj);
                return i;
            }

            template <typename Char>
            static std::basic_ostream<Char>&
            stream_out(std::basic_ostream<Char>& o, void* const* obj)
            {
                o << **reinterpret_cast<T* const*>(obj);
                return o;
            }
        };

        template <typename T, typename Small>
        struct streaming_base<T, Small, std::false_type>
        {
            template <typename Char>
            static std::basic_istream<Char>&
            stream_in (std::basic_istream<Char>& i, void** obj)
            {
                return i;
            }

            template <typename Char>
            static std::basic_ostream<Char>&
            stream_out(std::basic_ostream<Char>& o, void* const* obj)
            {
                return o;
            }
        };

        // static functions for small value-types
        template <typename Small>
        struct fxns;

        template <>
        struct fxns<std::true_type>
        {
            template<typename T, typename IArchive, typename OArchive, typename Char>
            struct type : public streaming_base<T, std::true_type>
            {
                static fxn_ptr_table<IArchive, OArchive, Char> *get_ptr()
                {
                    return detail::any::get_table<T>::
                        template get<IArchive, OArchive, Char>();
                }

                static boost::detail::sp_typeinfo const& get_type()
                {
                    return BOOST_SP_TYPEID(T);
                }
                static T & construct(void ** f)
                {
                    new (f) T;
                    return *reinterpret_cast<T *>(f);
                }

                static T & get(void **f)
                {
                    return *reinterpret_cast<T *>(f);
                }

                static T const & get(void *const*f)
                {
                    return *reinterpret_cast<T const *>(f);
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
                    *reinterpret_cast<T*>(dest) =
                        *reinterpret_cast<T const*>(src);
                }
                static bool equal_to(void* const* x, void* const* y)
                {
                    return (get(x) == get(y));
                }
            };
        };

        // static functions for big value-types (bigger than a void*)
        template <>
        struct fxns<std::false_type>
        {
            template<typename T, typename IArchive, typename OArchive, typename Char>
            struct type : public streaming_base<T, std::false_type>
            {
                static fxn_ptr_table<IArchive, OArchive, Char> *get_ptr()
                {
                    return detail::any::get_table<T>::
                        template get<IArchive, OArchive, Char>();
                }
                static boost::detail::sp_typeinfo const& get_type()
                {
                    return BOOST_SP_TYPEID(T);
                }
                static T & construct(void ** f)
                {
                    *f = new T;
                    return **reinterpret_cast<T **>(f);
                }
                static T & get(void **f)
                {
                    return **reinterpret_cast<T **>(f);
                }
                static T const & get(void *const*f)
                {
                    return **reinterpret_cast<T *const *>(f);
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
                    return (get(x) == get(y));
                }
            };
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IArchive, typename OArchive, typename Vtable, typename Char>
        struct fxn_ptr
          : fxn_ptr_table<IArchive, OArchive, Char>
        {
            typedef fxn_ptr_table<IArchive, OArchive, Char> base_type;

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

            virtual base_type * get_ptr()
            {
                return Vtable::get_ptr();
            }

            void save_object(void *const* object, OArchive & ar, unsigned)
            {
                ar & Vtable::get(object);
            }
            void load_object(void ** object, IArchive & ar, unsigned)
            {
                ar & Vtable::construct(object);
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & hpx::serialization::base_object<base_type>(*this);
            }
            HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(fxn_ptr);
        };

        template <typename Vtable, typename Char>
        struct fxn_ptr<void, void, Vtable, Char>
          : fxn_ptr_table<void, void, Char>
        {
            typedef fxn_ptr_table<void, void, Char> base_type;

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

            virtual base_type * get_ptr()
            {
                return Vtable::get_ptr();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct get_table
        {
            typedef std::integral_constant<bool, (sizeof(T) <= sizeof(void*))> is_small;

            template <typename IArchive, typename OArchive, typename Char>
            static fxn_ptr_table<IArchive, OArchive, Char>* get()
            {

                typedef
                    typename fxns<is_small>::
                        template type<T, IArchive, OArchive, Char>
                    fxn_type;

                typedef
                    fxn_ptr<IArchive, OArchive, fxn_type, Char>
                    fxn_ptr_type;

                static fxn_ptr_type static_table;

                return &static_table;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct empty
        {
            template <typename Archive>
            void serialize(Archive & ar, unsigned) {}
            bool operator==(empty const&) const
            {
                return false; // undefined
            }
            bool operator!=(empty const&) const
            {
                return false; // undefined
            }
        };

        template <typename Char>
        inline std::basic_istream<Char>&
        operator>> (std::basic_istream<Char>& i, empty&)
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
        inline std::basic_ostream<Char>&
        operator<< (std::basic_ostream<Char>& o, empty const&)
        {
            return o;
        }
    }} // namespace hpx::util::detail::any
}}  // namespace hpx::util

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename IArchive = serialization::input_archive,
        typename OArchive = serialization::output_archive,
        typename Char = char>
    class basic_any
    {
    public:
        // constructors
        basic_any() HPX_NOEXCEPT
          : table(detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>()),
            object(nullptr)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>()),
            object(nullptr)
        {
            assign(x);
        }

        template <typename T>
        explicit basic_any(T const& x)
          : table(detail::any::get_table<
                      typename util::decay<T>::type
                  >::template get<IArchive, OArchive, Char>()),
            object(nullptr)
        {
            typedef typename util::decay<T>::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(x);
            else
                object = new value_type(x);
        }

        // Move constructor
        basic_any(basic_any&& x) HPX_NOEXCEPT
          : table(x.table),
            object(x.object)
        {
            x.table = detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>();
            x.object = nullptr;
        }

        // Perfect forwarding of T
        template <typename T>
        explicit basic_any(T&& x,
            typename std::enable_if<
                !std::is_same<
                    basic_any,
                    typename util::decay<T>::type
                >::value>::type* = nullptr)
          : table(detail::any::get_table<
                      typename util::decay<T>::type
                  >::template get<IArchive, OArchive, Char>()),
            object(nullptr)
        {
            typedef typename util::decay<T>::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(std::forward<T>(x));
            else
                object = new value_type(std::forward<T>(x));
        }

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        // assignment
        basic_any& assign(basic_any const& x)
        {
            if (&x != this) {
                // are we copying between the same type?
                if (table == x.table) {
                    // if so, we can avoid reallocation
                    table->copy(&x.object, &object);
                }
                else {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

    public:
        // copy assignment operator
        basic_any& operator=(basic_any const& x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        // move assignement
        basic_any& operator=(basic_any&& rhs) HPX_NOEXCEPT
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
            if (&x == &y) // same object
            {
                return true;
            }

            if (x.table == y.table) // same type
            {
                return x.table->equal_to(&x.object, &y.object); // equal value?
            }

            return false;

        }

        template <typename T>
        friend bool operator==(basic_any const& b, T const& x)
        {
            typedef typename util::decay<T>::type value_type;

            if (b.type() == BOOST_SP_TYPEID(value_type)) // same type
            {
                return b.cast<value_type>() == x;
            }

            return false;
        }

        // inequality operator
        friend bool operator!=(basic_any const& x, basic_any const& y)
        {
            return !(x==y);
        }

        template <typename T>
        friend bool operator!=(basic_any const& b, T const& x)
        {
            return !(b==x);
        }

        // utility functions
        basic_any& swap(basic_any& x) HPX_NOEXCEPT
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        boost::detail::sp_typeinfo const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != BOOST_SP_TYPEID(T))
              throw bad_any_cast(type(), BOOST_SP_TYPEID(T));

            return detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

// implicit casting is disabled by default for compatibility with boost::any
#ifdef HPX_ANY_IMPLICIT_CASTING
        // automatic casting operator
        template <typename T>
        operator T const& () const { return cast<T>(); }
#endif // implicit casting

        bool empty() const HPX_NOEXCEPT
        {
            return table == detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>();
        }

        void reset()
        {
            if (!empty())
            {
                table->static_delete(&object);
                table = detail::any::get_table<detail::any::empty>::
                    template get<IArchive, OArchive, Char>();
                object = nullptr;
            }
        }

        // these functions have been added in the assumption that the embedded
        // type has a corresponding operator defined, which is completely safe
        // because hpx::util::any is used only in contexts where these operators
        // exist
        template <typename IArchive_, typename OArchive_, typename Char_>
        friend std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
            basic_any<IArchive_, OArchive_, Char_>& obj);

        template <typename IArchive_, typename OArchive_, typename Char_>
        friend std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
            basic_any<IArchive_, OArchive_, Char_> const& obj);

    private:

        friend class hpx::serialization::access;

        void load(IArchive &ar, const unsigned version)
        {
            bool is_empty;
            ar & is_empty;

            if (is_empty)
            {
                reset();
            }
            else
            {
                typename detail::any::fxn_ptr_table<
                        IArchive, OArchive, Char
                > *p = nullptr;
                ar >> hpx::serialization::detail::raw_ptr(p);
                table = p->get_ptr();
                delete p;
                table->load_object(&object, ar, version);
            }
        }

        void save(OArchive &ar, const unsigned version) const
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

    private: // types
        template <typename T, typename IArchive_, typename OArchive_, typename Char_>
        friend T* any_cast(basic_any<IArchive_, OArchive_, Char_> *) HPX_NOEXCEPT;

        // fields
        detail::any::fxn_ptr_table<IArchive, OArchive, Char>* table;
        void* object;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename IArchive_, typename OArchive_, typename Char_>
    std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
        basic_any<IArchive_, OArchive_, Char_>& obj)
    {
        return obj.table->stream_in(i, &obj.object);
    }

    template <typename IArchive_, typename OArchive_, typename Char_>
    std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
        basic_any<IArchive_, OArchive_, Char_> const& obj)
    {
        return obj.table->stream_out(o, &obj.object);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Char> // default is char
    class basic_any<void, void, Char>
    {
    public:
        // constructors
        basic_any() HPX_NOEXCEPT
          : table(detail::any::get_table<
                detail::any::empty>::template get<void, void, Char>()),
            object(nullptr)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<
                detail::any::empty>::template get<void, void, Char>()),
            object(nullptr)
        {
            assign(x);
        }

        template <typename T>
        explicit basic_any(T const& x)
          : table(detail::any::get_table<
                      typename util::decay<T>::type
                  >::template get<void, void, Char>()),
            object(nullptr)
        {
            typedef typename util::decay<T>::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(x);
            else
                object = new value_type(x);
        }

        // Move constructor
        basic_any(basic_any&& x) HPX_NOEXCEPT
          : table(x.table),
            object(x.object)
        {
            x.object = nullptr;
            x.table = detail::any::get_table<detail::any::empty>::
                template get<void, void, Char>();
        }

        // Perfect forwarding of T
        template <typename T>
        explicit basic_any(T&& x,
            typename std::enable_if<
                !std::is_same<
                    basic_any,
                    typename util::decay<T>::type
                >::value>::type* = nullptr)
          : table(detail::any::get_table<
                      typename util::decay<T>::type
                  >::template get<void, void, Char>()),
            object(nullptr)
        {
            typedef typename util::decay<T>::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(std::forward<T>(x));
            else
                object = new value_type(std::forward<T>(x));
        }

        ~basic_any()
        {
            table->static_delete(&object);
        }

    private:
        basic_any& assign(basic_any const& x)
        {
            if (&x != this) {
                // are we copying between the same type?
                if (table == x.table) {
                    // if so, we can avoid reallocation
                    table->copy(&x.object, &object);
                }
                else {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

    public:
        // copy assignment operator
        basic_any& operator=(basic_any const& x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        // move assignment
        basic_any& operator=(basic_any&& rhs)
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
            if (&x == &y) // same object
            {
                return true;
            }

            if (x.table == y.table) // same type
            {
                return x.table->equal_to(&x.object, &y.object); // equal value?
            }

            return false;
        }

        template <typename T>
        friend bool operator==(basic_any const& b, T const& x)
        {
            typedef typename util::decay<T>::type value_type;

            if (b.type() == BOOST_SP_TYPEID(value_type)) // same type
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
        basic_any& swap(basic_any& x) HPX_NOEXCEPT
        {
            std::swap(table, x.table);
            std::swap(object, x.object);
            return *this;
        }

        boost::detail::sp_typeinfo const& type() const
        {
            return table->get_type();
        }

        template <typename T>
        T const& cast() const
        {
            if (type() != BOOST_SP_TYPEID(T))
              throw bad_any_cast(type(), BOOST_SP_TYPEID(T));

            return hpx::util::detail::any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

// implicit casting is disabled by default for compatibility with boost::any
#ifdef HPX_ANY_IMPLICIT_CASTING
        // automatic casting operator
        template <typename T>
        operator T const& () const { return cast<T>(); }
#endif // implicit casting

        bool empty() const HPX_NOEXCEPT
        {
            return table ==
                detail::any::get_table<detail::any::empty>::
                    template get<void, void, Char>();
        }

        void reset()
        {
            if (!empty())
            {
                table->static_delete(&object);
                table = detail::any::get_table<detail::any::empty>::
                    template get<void, void, Char>();
                object = nullptr;
            }
        }

        // these functions have been added in the assumption that the embedded
        // type has a corresponding operator defined, which is completely safe
        // because hpx::util::any is used only in contexts where these operators
        // exist
        template <typename IArchive_, typename OArchive_, typename Char_>
        friend std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
            basic_any<IArchive_, OArchive_, Char_>& obj);

        template <typename IArchive_, typename OArchive_, typename Char_>
        friend std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
            basic_any<IArchive_, OArchive_, Char_> const& obj);

    private: // types
        template <typename T, typename IArchive_, typename OArchive_, typename Char_>
        friend T* any_cast(basic_any<IArchive_, OArchive_, Char_> *) HPX_NOEXCEPT;

        // fields
        detail::any::fxn_ptr_table<void, void, Char>* table;
        void* object;
    };
    ///////////////////////////////////////////////////////////////////////////

    template <typename IArchive, typename OArchive, typename Char>
    void swap(basic_any<IArchive, OArchive, Char>& lhs,
        basic_any<IArchive, OArchive, Char>& rhs) HPX_NOEXCEPT
    {
        lhs.swap(rhs);
    }

    // boost::any-like casting
    template <typename T, typename IArchive, typename OArchive, typename Char>
    inline T* any_cast (basic_any<IArchive, OArchive, Char>* operand) HPX_NOEXCEPT
    {
        if (operand && operand->type() == BOOST_SP_TYPEID(T)) {
            return hpx::util::detail::any::get_table<T>::is_small::value ?
                reinterpret_cast<T*>(reinterpret_cast<void*>(&operand->object)) :
                reinterpret_cast<T*>(reinterpret_cast<void*>(operand->object));
        }
        return nullptr;
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    inline T const* any_cast(basic_any<IArchive, OArchive,
        Char> const* operand) HPX_NOEXCEPT
    {
        return any_cast<T>(const_cast<basic_any<IArchive, OArchive, Char>*>(operand));
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    T any_cast(basic_any<IArchive, OArchive, Char>& operand)
    {
        typedef typename std::remove_reference<T>::type nonref;

        nonref* result = any_cast<nonref>(&operand);
        if(!result)
            boost::throw_exception(bad_any_cast(operand.type(), BOOST_SP_TYPEID(T)));
        return static_cast<T>(*result);
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    T const& any_cast(basic_any<IArchive, OArchive, Char> const& operand)
    {
        typedef typename std::remove_reference<T>::type nonref;

        return any_cast<nonref const&>(const_cast<basic_any<IArchive, OArchive,
            Char> &>(operand));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // backwards compatibility
    typedef basic_any<serialization::input_archive, serialization::output_archive,
        char> any;
    typedef basic_any<serialization::input_archive, serialization::output_archive,
        wchar_t> wany;

    typedef basic_any<void, void, char> any_nonser;
    typedef basic_any<void, void, wchar_t> wany_nonser;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct hash_binary_filter : serialization::binary_filter
        {
            explicit hash_binary_filter(std::size_t seed = 0)
              : hash(seed)
            {}

            // compression API
            void set_max_length(std::size_t size)
            {}
            void save(void const* src, std::size_t src_count)
            {
                char const* data = static_cast<char const*>(src);
                boost::hash_range(hash, data, data + src_count);
            }
            bool flush(void* dst, std::size_t dst_count,
                std::size_t& written)
            {
                return true;
            }

            // decompression API
            std::size_t init_data(char const* buffer,
                std::size_t size, std::size_t buffer_size)
            {
                return 0;
            }
            void load(void* dst, std::size_t dst_count)
            {}

            template <class T> void serialize(T&, unsigned){}
            HPX_SERIALIZATION_POLYMORPHIC(hash_binary_filter);

            std::size_t hash;
        };
    }

    struct hash_any
    {
        template <typename Char>
        size_t operator()(const basic_any<
                serialization::input_archive,
                serialization::output_archive,
                Char
            > &elem) const
        {
            detail::hash_binary_filter hasher;

            {
                std::vector<char> data;
                serialization::output_archive ar (
                        data, 0U, ~0U, nullptr, &hasher);
                ar << elem;
            }  // let archive go out of scope

            return hasher.hash;
        }
    };
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#if BOOST_WORKAROUND(HPX_MSVC, >= 1400)
# pragma warning(pop)
#endif

#endif
