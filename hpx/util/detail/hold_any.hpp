/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang
    Copyright (c) 2007-2011 Hartmut Kaiser
    Copyright (c) Christopher Diggins 2005
    Copyright (c) Pablo Aguilar 2005
    Copyright (c) Kevlin Henney 2001

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    The class hpx::util::detail::hold_any is built based on boost::spirit::hold_any class.
    It adds support for serialization.
==============================================================================*/
#ifndef HPX_UTIL_HOLD_ANY_HPP
#define HPX_UTIL_HOLD_ANY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/type_traits/is_reference.hpp>
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

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>

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

    namespace detail { namespace hold_any
    {
        template <typename T>
        struct get_table;

        // serializable function pointer table
        template <typename Char, typename IArchive, typename OArchive>
        struct fxn_ptr_table
        {
            virtual fxn_ptr_table * get_ptr() = 0;
            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*move)(void* const*, void**);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&, void* const*);

            virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
            virtual void load_object(void **, IArchive & ar, unsigned) = 0;

            template <typename Archive>
            void serialize(Archive & ar, unsigned) {}
        };

        // function pointer table
        template <typename Char>
        struct fxn_ptr_table<Char, void, void>
        {
            virtual fxn_ptr_table * get_ptr() = 0;
            boost::detail::sp_typeinfo const& (*get_type)();
            void (*static_delete)(void**);
            void (*destruct)(void**);
            void (*clone)(void* const*, void**);
            void (*move)(void* const*, void**);
            std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
            std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&, void* const*);
        };

        // static functions for small value-types
        template <typename Small>
        struct fxns;

        template <>
        struct fxns<boost::mpl::true_>
        {
            template<typename T, typename Char
            , typename IArchive, typename OArchive
            >
            struct type
            {
                static fxn_ptr_table<Char, IArchive, OArchive> *get_ptr()
                {
                    return detail::hold_any::get_table<T>::
                        template get<Char, IArchive, OArchive>();
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
                static void move(void* const* src, void** dest)
                {
                    *reinterpret_cast<T*>(dest) =
                        *reinterpret_cast<T const*>(src);
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
            template<typename T, typename Char
            , typename IArchive, typename OArchive
            >
            struct type
            {
                static fxn_ptr_table<Char, IArchive, OArchive> *get_ptr()
                {
                    return detail::hold_any::get_table<T>::
                        template get<Char, IArchive, OArchive>();
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
                static void move(void* const* src, void** dest)
                {
                    **reinterpret_cast<T**>(dest) =
                        **reinterpret_cast<T* const*>(src);
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
        template <typename Char, typename IArchive, typename OArchive, typename Vtable>
        struct fxn_ptr
          : fxn_ptr_table<Char, IArchive, OArchive>
        {
            typedef fxn_ptr_table<Char, IArchive, OArchive> base_type;

            fxn_ptr()
            {
                base_type::get_type = Vtable::get_type;
                base_type::static_delete = Vtable::static_delete;
                base_type::destruct = Vtable::destruct;
                base_type::clone = Vtable::clone;
                base_type::move = Vtable::move;
                base_type::stream_in = Vtable::stream_in;
                base_type::stream_out = Vtable::stream_out;

#if 0
                // make sure the global gets instantiated;
                hpx::actions::detail::guid_initialization<fxn_ptr>();
#endif
            }

            virtual base_type * get_ptr()
            {
                return Vtable::get_ptr();
            }

#if 0
            static void register_base()
            {
                util::void_cast_register_nonvirt<fxn_ptr, base_type>();
            }
#endif

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

        template <typename Char, typename Vtable>
        struct fxn_ptr<Char, void, void, Vtable>
          : fxn_ptr_table<Char, void, void>
        {
            typedef fxn_ptr_table<Char, void, void> base_type;

            fxn_ptr()
            {
                base_type::get_type = Vtable::get_type;
                base_type::static_delete = Vtable::static_delete;
                base_type::destruct = Vtable::destruct;
                base_type::clone = Vtable::clone;
                base_type::move = Vtable::move;
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

            template <typename Char, typename IArchive, typename OArchive>
            static fxn_ptr_table<Char, IArchive, OArchive>* get()
            {

                typedef
                    typename fxns<is_small>::
                        template type<T, Char, IArchive, OArchive>
                    fxn_type;

                typedef
                    fxn_ptr<Char, IArchive, OArchive, fxn_type>
                    fxn_ptr_type;

                static fxn_ptr_type static_table;

                return &static_table;

#if 0
                static fxn_ptr_table<Char, IArchive, OArchive> static_table =
                {
                    fxns<is_small>::template type<T, Char>::get_type,
                    fxns<is_small>::template type<T, Char>::static_delete,
                    fxns<is_small>::template type<T, Char>::destruct,
                    fxns<is_small>::template type<T, Char>::clone,
                    fxns<is_small>::template type<T, Char>::move,
                    fxns<is_small>::template type<T, Char>::stream_in,
                    fxns<is_small>::template type<T, Char>::stream_out
                };
                return &static_table;
#endif
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct empty
        {
            template <typename Archive>
            void serialize(Archive & ar, unsigned) {}
        };

        template <typename Char>
        inline std::basic_istream<Char>&
        operator>> (std::basic_istream<Char>& i, empty&)
        {
            // If this assertion fires you tried to insert from a std istream
            // into an empty hold_any instance. This simply can't work, because
            // there is no way to figure out what type to extract from the
            // stream.
            // The only way to make this work is to assign an arbitrary
            // value of the required type to the hold_any instance you want to
            // stream to. This assignment has to be executed before the actual
            // call to the operator>>().
            BOOST_ASSERT(false &&
                "Tried to insert from a std istream into an empty "
                "hold_any instance");
            return i;
        }

        template <typename Char>
        inline std::basic_ostream<Char>&
        operator<< (std::basic_ostream<Char>& o, empty const&)
        {
            return o;
        }
    }} // namespace hpx::util::detail::hold_any

    ///////////////////////////////////////////////////////////////////////////
    template <typename Char
        , typename IArchive = portable_binary_iarchive
        , typename OArchive = portable_binary_oarchive>
    class basic_hold_any
    {
    public:
        // constructors
        template <typename T>
        explicit basic_hold_any(T const& x)
          : table(detail::hold_any::get_table<T>::
                template get<Char, IArchive, OArchive>()), object(0)
        {
            if (hpx::util::detail::hold_any::get_table<T>::is_small::value)
                new (&object) T(x);
            else
                object = new T(x);
        }

        basic_hold_any()
          : table(detail::hold_any::get_table<detail::hold_any::empty>::
                template get<Char, IArchive, OArchive>()),
            object(0)
        {
        }

        basic_hold_any(basic_hold_any const& x)
          : table(detail::hold_any::get_table<detail::hold_any::empty>::
                template get<Char, IArchive, OArchive>()),
            object(0)
        {
            assign(x);
        }

        ~basic_hold_any()
        {
            table->static_delete(&object);
        }

        // assignment
        basic_hold_any& assign(basic_hold_any const& x)
        {
            if (&x != this) {
                // are we copying between the same type?
                if (table == x.table) {
                    // if so, we can avoid reallocation
                    table->move(&x.object, &object);
                }
                else {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

        template <typename T>
        basic_hold_any& assign(T const& x)
        {
            // are we copying between the same type?
            detail::hold_any::fxn_ptr_table<Char, IArchive, OArchive>* x_table =
                detail::hold_any::get_table<T>::template get<Char, IArchive, OArchive>();

            if (table == x_table) {
            // if so, we can avoid deallocating and re-use memory
                table->destruct(&object);    // first destruct the old content
                if (hpx::util::detail::hold_any::get_table<T>::is_small::value) {
                    // create copy on-top of object pointer itself
                    new (&object) T(x);
                }
                else {
                    // create copy on-top of old version
                    new (object) T(x);
                }
            }
            else {
                if (hpx::util::detail::hold_any::get_table<T>::is_small::value) {
                    // create copy on-top of object pointer itself
                    table->destruct(&object); // first destruct the old content
                    new (&object) T(x);
                }
                else {
                    reset();                  // first delete the old content
                    object = new T(x);
                }
                table = x_table;      // update table pointer
            }
            return *this;
        }

        // assignment operator
        template <typename T>
        basic_hold_any& operator=(T const& x)
        {
            return assign(x);
        }

        // utility functions
        basic_hold_any& swap(basic_hold_any& x)
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

            return hpx::util::detail::hold_any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

// implicit casting is disabled by default for compatibility with boost::any
#ifdef HPX_ANY_IMPLICIT_CASTING
        // automatic casting operator
        template <typename T>
        operator T const& () const { return cast<T>(); }
#endif // implicit casting

        bool empty() const
        {
            return table == detail::hold_any::get_table<detail::hold_any::empty>::
                template get<Char, IArchive, OArchive>();
        }

        void reset()
        {
            if (!empty())
            {
                table->static_delete(&object);
                table = detail::hold_any::get_table<detail::hold_any::empty>::
                    template get<Char, IArchive, OArchive>();
                object = 0;
            }
        }

        // these functions have been added in the assumption that the embedded
        // type has a corresponding operator defined, which is completely safe
        // because spirit::hold_any is used only in contexts where these operators
        // do exist
        template <typename Char_, typename IArchive_, typename OArchive_>
        friend inline std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i,
            basic_hold_any<Char_, IArchive_, OArchive_>& obj)
        {
            return obj.table->stream_in(i, &obj.object);
        }

        template <typename Char_, typename IArchive_, typename OArchive_>
        friend inline std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o,
            basic_hold_any<Char_, IArchive_, OArchive_> const& obj)
        {
            return obj.table->stream_out(o, &obj.object);
        }

    private:

        friend class boost::serialization::access;

        void load(IArchive &ar, const unsigned version)
        {
            bool is_empty;
            ar & is_empty;

            if(is_empty)
            {
                reset();
            }
            else
            {
                typename detail::hold_any::fxn_ptr_table<Char, IArchive, OArchive> *p = 0;
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
            if(!empty())
            {
                ar << table;
                table->save_object(&object, ar, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
    private: // types
        template <typename T, typename Char_
            , typename IArchive_, typename OArchive_
        >
        friend T* any_cast(basic_hold_any<Char_, IArchive_, OArchive_> *);
#else
    public: // types (public so any_cast can be non-friend)
#endif
        // fields
        detail::hold_any::fxn_ptr_table<Char, IArchive, OArchive>* table;
        void* object;
    };


    ///////////////////////////////////////////////////////////////////////////
    template <typename Char>
    class basic_hold_any<Char, void, void>
    {
    public:
        // constructors
        template <typename T>
        explicit basic_hold_any(T const& x)
          : table(hpx::util::detail::hold_any::get_table<T>::template get<Char, void, void>()), object(0)
        {
            if (hpx::util::detail::hold_any::get_table<T>::is_small::value)
                new (&object) T(x);
            else
                object = new T(x);
        }

        basic_hold_any()
          : table(hpx::util::detail::hold_any::get_table<hpx::util::detail::hold_any::empty>::template get<Char, void, void>()),
            object(0)
        {
        }

        basic_hold_any(basic_hold_any const& x)
          : table(hpx::util::detail::hold_any::get_table<hpx::util::detail::hold_any::empty>::template get<Char, void, void>()),
            object(0)
        {
            assign(x);
        }

        ~basic_hold_any()
        {
            table->static_delete(&object);
        }

        // assignment
        basic_hold_any& assign(basic_hold_any const& x)
        {
            if (&x != this) {
                // are we copying between the same type?
                if (table == x.table) {
                    // if so, we can avoid reallocation
                    table->move(&x.object, &object);
                }
                else {
                    reset();
                    x.table->clone(&x.object, &object);
                    table = x.table;
                }
            }
            return *this;
        }

        template <typename T>
        basic_hold_any& assign(T const& x)
        {
            // are we copying between the same type?
            hpx::util::detail::hold_any::fxn_ptr_table<Char, void, void>* x_table =
                hpx::util::detail::hold_any::get_table<T>::template get<Char, void, void>();
            if (table == x_table) {
            // if so, we can avoid deallocating and re-use memory
                table->destruct(&object);    // first destruct the old content
                if (hpx::util::detail::hold_any::get_table<T>::is_small::value) {
                    // create copy on-top of object pointer itself
                    new (&object) T(x);
                }
                else {
                    // create copy on-top of old version
                    new (object) T(x);
                }
            }
            else {
                if (hpx::util::detail::hold_any::get_table<T>::is_small::value) {
                    // create copy on-top of object pointer itself
                    table->destruct(&object); // first destruct the old content
                    new (&object) T(x);
                }
                else {
                    reset();                  // first delete the old content
                    object = new T(x);
                }
                table = x_table;      // update table pointer
            }
            return *this;
        }

        // assignment operator
        template <typename T>
        basic_hold_any& operator=(T const& x)
        {
            return assign(x);
        }

        // utility functions
        basic_hold_any& swap(basic_hold_any& x)
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

            return hpx::util::detail::hold_any::get_table<T>::is_small::value ?
                *reinterpret_cast<T const*>(&object) :
                *reinterpret_cast<T const*>(object);
        }

// implicit casting is disabled by default for compatibility with boost::any
#ifdef BOOST_SPIRIT_ANY_IMPLICIT_CASTING
        // automatic casting operator
        template <typename T>
        operator T const& () const { return cast<T>(); }
#endif // implicit casting

        bool empty() const
        {
            return table == hpx::util::detail::hold_any::get_table<hpx::util::detail::hold_any::empty>::template get<Char, void, void>();
        }

        void reset()
        {
            if (!empty())
            {
                table->static_delete(&object);
                table = hpx::util::detail::hold_any::get_table<hpx::util::detail::hold_any::empty>::template get<Char, void, void>();
                object = 0;
            }
        }

    // these functions have been added in the assumption that the embedded
    // type has a corresponding operator defined, which is completely safe
    // because spirit::hold_any is used only in contexts where these operators
    // do exist
        template <typename Char_>
        friend inline std::basic_istream<Char_>&
        operator>> (std::basic_istream<Char_>& i, basic_hold_any<Char_, void, void>& obj)
        {
            return obj.table->stream_in(i, &obj.object);
        }

        template <typename Char_>
        friend inline std::basic_ostream<Char_>&
        operator<< (std::basic_ostream<Char_>& o, basic_hold_any<Char_, void, void> const& obj)
        {
            return obj.table->stream_out(o, &obj.object);
        }

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
    private: // types
        template <typename T, typename Char_>
        friend T* any_cast_nonser(basic_hold_any<Char_, void, void> *);
#else
    public: // types (public so any_cast can be non-friend)
#endif
        // fields
        hpx::util::detail::hold_any::fxn_ptr_table<Char, void, void>* table;
        void* object;
    };
    ///////////////////////////////////////////////////////////////////////////

    // boost::any-like casting
    template <typename T, typename Char, typename IArchive, typename OArchive>
    inline T* any_cast (basic_hold_any<Char, IArchive, OArchive>* operand)
    {
        if (operand && operand->type() == BOOST_SP_TYPEID(T)) {
            return hpx::util::detail::hold_any::get_table<T>::is_small::value ?
                reinterpret_cast<T*>(&operand->object) :
                reinterpret_cast<T*>(operand->object);
        }
        return 0;
    }

    template <typename T, typename Char, typename IArchive, typename OArchive>
    inline T const* any_cast(basic_hold_any<Char, IArchive, OArchive> const* operand)
    {
        return any_cast<T>(const_cast<basic_hold_any<Char, IArchive, OArchive>*>(operand));
    }

    template <typename T, typename Char, typename IArchive, typename OArchive>
    T any_cast(basic_hold_any<Char, IArchive, OArchive>& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME hpx::util::detail::remove_reference<T>::type nonref;

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
        return *result;
    }

    template <typename T, typename Char, typename IArchive, typename OArchive>
    T const& any_cast(basic_hold_any<Char, IArchive, OArchive> const& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME hpx::util::detail::remove_reference<T>::type nonref;

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // The comment in the above version of 'any_cast' explains when this
        // assert is fired and what to do.
        BOOST_STATIC_ASSERT(!is_reference<nonref>::value);
#endif

        return any_cast<nonref const&>(const_cast<basic_hold_any<Char, IArchive, OArchive> &>(operand));
    }

    ///////////////////////////////////////////////////////////////////////////

    // boost::any-like casting non-serialized versions
    template <typename T, typename Char>
    inline T* any_cast_nonser (basic_hold_any<Char, void, void>* operand)
    {
        if (operand && operand->type() == BOOST_SP_TYPEID(T)) {
            return hpx::util::detail::hold_any::get_table<T>::is_small::value ?
                reinterpret_cast<T*>(&operand->object) :
                reinterpret_cast<T*>(operand->object);
        }
        return 0;
    }

    template <typename T, typename Char>
    inline T const* any_cast_nonser(basic_hold_any<Char, void, void> const* operand)
    {
        return any_cast_nonser<T>(const_cast<basic_hold_any<Char, void, void>*>(operand));
    }

    template <typename T, typename Char>
    T any_cast_nonser(basic_hold_any<Char, void, void>& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME hpx::util::detail::remove_reference<T>::type nonref;

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // If 'nonref' is still reference type, it means the user has not
        // specialized 'remove_reference'.

        // Please use BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION macro
        // to generate specialization of remove_reference for your class
        // See type traits library documentation for details
        BOOST_STATIC_ASSERT(!is_reference<nonref>::value);
#endif

        nonref* result = any_cast_nonser<nonref>(&operand);
        if(!result)
            boost::throw_exception(bad_any_cast(operand.type(), BOOST_SP_TYPEID(T)));
        return *result;
    }

    template <typename T, typename Char>
    T const& any_cast_nonser(basic_hold_any<Char, void, void> const& operand)
    {
        typedef BOOST_DEDUCED_TYPENAME hpx::util::detail::remove_reference<T>::type nonref;

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // The comment in the above version of 'any_cast' explains when this
        // assert is fired and what to do.
        BOOST_STATIC_ASSERT(!is_reference<nonref>::value);
#endif

        return any_cast_nonser<nonref const&>(const_cast<basic_hold_any<Char, void, void> &>(operand));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // backwards compatibility
    typedef basic_hold_any<char> hold_any;
    typedef basic_hold_any<wchar_t> whold_any;

    typedef basic_hold_any<char, void, void> hold_any_nonser;
    typedef basic_hold_any<wchar_t, void, void> whold_any_nonser;

}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

#endif
