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

#if defined(_MSC_VER)
#pragma once
#endif

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/move.hpp>

#include <boost/config.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/throw_exception.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/assert.hpp>
#include <boost/detail/sp_typeinfo.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/type_traits/decay.hpp>

#include <stdexcept>
#include <typeinfo>
#include <algorithm>
#include <iosfwd>

///////////////////////////////////////////////////////////////////////////////
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
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
            virtual fxn_ptr_table * get_ptr() = 0;
            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*copy)(void* const*, void**);
            bool (*equal_to)(void* const*, void* const*);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&, void* const*);

            virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
            virtual void load_object(void **, IArchive & ar, unsigned) = 0;

            template <typename Archive>
            void serialize(Archive & ar, unsigned) {}
        };

        // function pointer table
        template <typename Char>
        struct fxn_ptr_table<void, void, Char>
        {
            virtual fxn_ptr_table * get_ptr() = 0;
            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*copy)(void* const*, void**);
            bool (*equal_to)(void* const*, void* const*);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&, void* const*);
        };

        // static functions for small value-types
        template <typename Small>
        struct fxns;

        template <>
        struct fxns<boost::mpl::true_>
        {
            template<typename T, typename IArchive, typename OArchive, typename Char>
            struct type
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
                static std::basic_istream<Char>&
                stream_in (std::basic_istream<Char>& i, void** obj)
                {
                    i >> *reinterpret_cast<T*>(obj);
                    return i;
                }
                static std::basic_ostream<Char>&
                stream_out(std::basic_ostream<Char>& o, void* const* obj)
                {
                    o << *reinterpret_cast<T const*>(obj);
                    return o;
                }
            };
        };

        // static functions for big value-types (bigger than a void*)
        template <>
        struct fxns<boost::mpl::false_>
        {
            template<typename T, typename IArchive, typename OArchive, typename Char>
            struct type
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
                static std::basic_istream<Char>&
                stream_in(std::basic_istream<Char>& i, void** obj)
                {
                    i >> **reinterpret_cast<T**>(obj);
                    return i;
                }
                static std::basic_ostream<Char>&
                stream_out(std::basic_ostream<Char>& o, void* const* obj)
                {
                    o << **reinterpret_cast<T* const*>(obj);
                    return o;
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

                // make sure the global gets instantiated;
                hpx::actions::detail::guid_initialization<fxn_ptr>();
            }

            virtual base_type * get_ptr()
            {
                return Vtable::get_ptr();
            }

            static void register_base()
            {
                util::void_cast_register_nonvirt<fxn_ptr, base_type>();
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
                ar & boost::serialization::base_object<base_type>(*this);
            }
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
            typedef boost::mpl::bool_<(sizeof(T) <= sizeof(void*))> is_small;

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
            BOOST_ASSERT(false &&
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

///////////////////////////////////////////////////////////////////////////////
// Make sure any serialization is properly initialized
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename IArchive, typename OArchive, typename Vtable, typename Char>)
  , (hpx::util::detail::any::fxn_ptr<IArchive, OArchive, Vtable, Char>)
)

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename IArchive = portable_binary_iarchive,
        typename OArchive = portable_binary_oarchive,
        typename Char = char>
    class basic_any
    {
    private:
        // Mark this class copyable and movable
        BOOST_COPYABLE_AND_MOVABLE(basic_any)

    public:
        // constructors
        basic_any() BOOST_NOEXCEPT
          : table(detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>()),
            object(0)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>()),
            object(0)
        {
            assign(x);
        }

        template <typename T>
        basic_any(T const& x)
          : table(detail::any::get_table<
                      typename boost::remove_const<
                          typename util::detail::remove_reference<T>::type
                      >::type
                  >::template get<IArchive, OArchive, Char>()),
            object(0)
        {
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(x);
            else
                object = new value_type(x);
        }

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        // Move constructor
        basic_any(basic_any&& x) BOOST_NOEXCEPT
          : table(x.table),
            object(x.object)
        {
            x.table = detail::any::get_table<detail::any::empty>::
                template get<IArchive, OArchive, Char>();
            x.object = 0;
        }

        // Perfect forwarding of T
        template <typename T>
        basic_any(T&& x, typename boost::disable_if<boost::is_same<basic_any&, T> >::type* = 0)
          : table(detail::any::get_table<
                      typename boost::remove_const<
                          typename util::detail::remove_reference<T>::type
                      >::type
                  >::template get<IArchive, OArchive, Char>()),
            object(0)
        {
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(boost::forward<T>(x));
            else
                object = new value_type(boost::forward<T>(x));
        }
#endif

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
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
        // copy assignment operator
        basic_any& operator=(basic_any x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        template <typename T>
        basic_any& operator=(T const& rhs)
        {
            basic_any(rhs).swap(*this);
            return *this;
        }
#else
        // copy assignment operator
        basic_any& operator=(basic_any const& x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        // move assignement
        basic_any& operator=(basic_any&& rhs) BOOST_NOEXCEPT
        {
            rhs.swap(*this);
            basic_any().swap(rhs);
            return *this;
        }

        // Perfect forwarding of T
        template <typename T>
        basic_any& operator=(T&& rhs)
        {
            basic_any(boost::forward<T>(rhs)).swap(*this);
            return *this;
        }
#endif

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
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

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
        basic_any& swap(basic_any& x) BOOST_NOEXCEPT
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

        bool empty() const BOOST_NOEXCEPT
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
                object = 0;
            }
        }

        // these functions have been added in the assumption that the embedded
        // type has a corresponding operator defined, which is completely safe
        // because hpx::uti::any is used only in contexts where these operators
        // do exist
        template <typename IArchive_, typename OArchive_, typename Char_>
        friend inline std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
            basic_any<IArchive_, OArchive_, Char_>& obj)
        {
            return obj.table->stream_in(i, &obj.object);
        }

        template <typename IArchive_, typename OArchive_, typename Char_>
        friend inline std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
            basic_any<IArchive_, OArchive_, Char_> const& obj)
        {
            return obj.table->stream_out(o, &obj.object);
        }

    private:

        friend class boost::serialization::access;

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
                typename detail::any::fxn_ptr_table<IArchive, OArchive, Char> *p = 0;
                ar >> p;
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
                ar << table;
                table->save_object(&object, ar, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
    private: // types
        template <typename T, typename IArchive_, typename OArchive_, typename Char_>
        friend T* any_cast(basic_any<IArchive_, OArchive_, Char_> *) BOOST_NOEXCEPT;
#else
    public: // types (public so any_cast can be non-friend)
#endif
        // fields
        detail::any::fxn_ptr_table<IArchive, OArchive, Char>* table;
        void* object;
    };


    ///////////////////////////////////////////////////////////////////////////
    template <typename Char> // default is char
    class basic_any<void, void, Char>
    {
    private:
        // Mark this class copyable and movable
        BOOST_COPYABLE_AND_MOVABLE(basic_any)

    public:
        // constructors
        basic_any() BOOST_NOEXCEPT
          : table(detail::any::get_table<
                detail::any::empty>::template get<void, void, Char>()),
            object(0)
        {
        }

        basic_any(basic_any const& x)
          : table(detail::any::get_table<
                detail::any::empty>::template get<void, void, Char>()),
            object(0)
        {
            assign(x);
        }

        template <typename T>
        basic_any(T const& x)
          : table(detail::any::get_table<
                      typename boost::remove_const<
                          typename util::detail::remove_reference<T>::type
                      >::type
                  >::template get<void, void, Char>()),
            object(0)
        {
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(x);
            else
                object = new value_type(x);
        }

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        // Move constructor
        basic_any(basic_any&& x) BOOST_NOEXCEPT
          : table(x.table),
            object(x.object)
        {
            x.object = 0;
            x.table = detail::any::get_table<detail::any::empty>::
                template get<void, void, Char>();
        }

        // Perfect forwarding of T
        template <typename T>
        basic_any(T&& x, typename boost::disable_if<boost::is_same<basic_any&, T> >::type* = 0)
          : table(detail::any::get_table<
                      typename boost::remove_const<
                          typename util::detail::remove_reference<T>::type
                      >::type
                  >::template get<void, void, Char>()),
            object(0)
        {
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

            if (detail::any::get_table<value_type>::is_small::value)
                new (&object) value_type(boost::forward<T>(x));
            else
                object = new value_type(boost::forward<T>(x));
        }
#endif

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
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
        // copy assignment operator
        basic_any& operator=(basic_any x)
        {
            basic_any(x).swap(*this);
            return *this;
        }

        template <typename T>
        basic_any& operator=(T const& rhs)
        {
            basic_any(rhs).swap(*this);
            return *this;
        }
#else
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
            basic_any(boost::forward<T>(rhs)).swap(*this);
            return *this;
        }
#endif

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
            typedef typename boost::remove_const<
                typename util::detail::remove_reference<T>::type
            >::type value_type;

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
        basic_any& swap(basic_any& x) BOOST_NOEXCEPT
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

        bool empty() const BOOST_NOEXCEPT
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
                object = 0;
            }
        }

    // these functions have been added in the assumption that the embedded
    // type has a corresponding operator defined, which is completely safe
    // because hpx::util::any is used only in contexts where these operators
    // do exist
        template <typename Char_>
        friend inline std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
            basic_any<void, void, Char_>& obj)
        {
            return obj.table->stream_in(i, &obj.object);
        }

        template <typename Char_>
        friend inline std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
            basic_any<void, void, Char_> const& obj)
        {
            return obj.table->stream_out(o, &obj.object);
        }

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
    private: // types
        template <typename T, typename IArchive_, typename OArchive_, typename Char_>
        friend T* any_cast(basic_any<IArchive_, OArchive_, Char_> *) BOOST_NOEXCEPT;
#else
    public: // types (public so any_cast can be non-friend)
#endif
        // fields
        detail::any::fxn_ptr_table<void, void, Char>* table;
        void* object;
    };
    ///////////////////////////////////////////////////////////////////////////

    template <typename IArchive, typename OArchive, typename Char>
    void swap(basic_any<IArchive, OArchive, Char>& lhs,
        basic_any<IArchive, OArchive, Char>& rhs) BOOST_NOEXCEPT
    {
        lhs.swap(rhs);
    }

    // boost::any-like casting
    template <typename T, typename IArchive, typename OArchive, typename Char>
    inline T* any_cast (basic_any<IArchive, OArchive, Char>* operand) BOOST_NOEXCEPT
    {
        if (operand && operand->type() == BOOST_SP_TYPEID(T)) {
            return hpx::util::detail::any::get_table<T>::is_small::value ?
                reinterpret_cast<T*>(reinterpret_cast<void*>(&operand->object)) :
                reinterpret_cast<T*>(reinterpret_cast<void*>(operand->object));
        }
        return 0;
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    inline T const* any_cast(basic_any<IArchive, OArchive, Char> const* operand) BOOST_NOEXCEPT
    {
        return any_cast<T>(const_cast<basic_any<IArchive, OArchive, Char>*>(operand));
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    T any_cast(basic_any<IArchive, OArchive, Char>& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME detail::remove_reference<T>::type nonref;

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // If 'nonref' is still reference type, it means the user has not
        // specialized 'remove_reference'.

        // Please use BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION macro
        // to generate specialization of remove_reference for your class
        // See type traits library documentation for details
        BOOST_STATIC_ASSERT(!is_reference<nonref>::value);
#endif

        nonref* result = any_cast<nonref>(&operand);
        if(!result)
            boost::throw_exception(bad_any_cast(operand.type(), BOOST_SP_TYPEID(T)));
        return static_cast<T>(*result);
    }

    template <typename T, typename IArchive, typename OArchive, typename Char>
    T const& any_cast(basic_any<IArchive, OArchive, Char> const& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME detail::remove_reference<T>::type nonref;

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // The comment in the above version of 'any_cast' explains when this
        // assert is fired and what to do.
        BOOST_STATIC_ASSERT(!is_reference<nonref>::value);
#endif

        return any_cast<nonref const&>(const_cast<basic_any<IArchive, OArchive, Char> &>(operand));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // backwards compatibility
    typedef basic_any<portable_binary_iarchive, portable_binary_oarchive> any;
    typedef basic_any<portable_binary_iarchive, portable_binary_oarchive, wchar_t> wany;

    typedef basic_any<void, void, char> any_nonser;
    typedef basic_any<void, void, wchar_t> wany_nonser;

}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

#endif
