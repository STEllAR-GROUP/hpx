#ifndef PORTABLE_BINARY_IARCHIVE_HPP
#define PORTABLE_BINARY_IARCHIVE_HPP

#include <boost/version.hpp>
#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_integral.hpp>

#if !defined(HPX_USE_PORTABLE_ARCHIVES) || HPX_USE_PORTABLE_ARCHIVES == 0
#include <boost/archive/binary_iarchive.hpp>

namespace hpx { namespace util
{
    typedef boost::archive::binary_iarchive portable_binary_iarchive;
}}

#else

// MS compatible compilers support #pragma once
#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// portable_binary_iarchive.hpp

// (C) Copyright 2002-7 Robert Ramey - http://www.rrsd.com .
// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/common_iarchive.hpp>
#include <boost/archive/shared_ptr_helper.hpp>
#include <boost/archive/detail/register_archive.hpp>
#if BOOST_VERSION >= 104400
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/collection_size_type.hpp>
#endif

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>
#include <hpx/util/basic_binary_iprimitive.hpp>

#include <limits>

namespace hpx { namespace util
{

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// exception to be thrown if integer read from archive doesn't fit
// variable being loaded
class HPX_ALWAYS_EXPORT portable_binary_iarchive_exception :
    public virtual boost::archive::archive_exception
{
public:
    enum exception_code {
        incompatible_integer_size
    };
    portable_binary_iarchive_exception(exception_code c = incompatible_integer_size )
      : boost::archive::archive_exception(
            static_cast<boost::archive::archive_exception::exception_code>(c))
    {}
    virtual const char *what() const throw()
    {
        const char *msg = "programmer error";
        switch (static_cast<exception_code>(code)) {
        case incompatible_integer_size:
            msg = "integer cannot be represented";
        default:
            boost::archive::archive_exception::what();
        }
        return msg;
    }
};

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// "Portable" input binary archive.  It addresses integer size and endianness so
// that binary archives can be passed across systems. Note:floating point types
// are passed through as is.
#if defined(BOOST_MSVC) || defined(BOOST_INTEL_WIN)
#define HPX_SERIALIZATION_EXPORT
#else
#define HPX_SERIALIZATION_EXPORT HPX_ALWAYS_EXPORT
#endif

class HPX_SERIALIZATION_EXPORT portable_binary_iarchive :
    public hpx::util::basic_binary_iprimitive<
        portable_binary_iarchive
    >,
    public boost::archive::detail::common_iarchive<
        portable_binary_iarchive
    >,
    public boost::archive::detail::shared_ptr_helper
{
    typedef hpx::util::basic_binary_iprimitive<
        portable_binary_iarchive
    > primitive_base_t;
    typedef boost::archive::detail::common_iarchive<
        portable_binary_iarchive
    > archive_base_t;
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
    friend archive_base_t;
    friend primitive_base_t; // since with override load below
    friend class boost::archive::detail::interface_iarchive<
        portable_binary_iarchive>;
    friend class boost::archive::load_access;
protected:
#endif

    HPX_ALWAYS_EXPORT void load_impl(boost::int64_t& l, char const maxsize);
    HPX_ALWAYS_EXPORT void load_impl(boost::uint64_t& l, char const maxsize);

    // default fall through for any types not specified here
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
    template <typename T>
    void load_integral(T& t, boost::mpl::false_)
    {
        boost::int64_t l = 0;
        load_impl(l, sizeof(T));
        t = static_cast<T>(l);      // use cast to avoid compile time warning
    }

    template <typename T>
    void load_integral(T& t, boost::mpl::true_)
    {
        boost::uint64_t l = 0;
        load_impl(l, sizeof(T));
        t = static_cast<T>(l);      // use cast to avoid compile time warning
    }
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

    template <typename T>
    void load(T& t, typename boost::enable_if<boost::is_integral<T> >::type* = 0)
    {
        load_integral(t, typename boost::is_unsigned<T>::type());
    }

    template <typename T>
    void load(T& t, typename boost::disable_if<boost::is_integral<T> >::type* = 0)
    {
        this->primitive_base_t::load(t);
    }

    void load(std::string& t) {
        this->primitive_base_t::load(t);
    }
#if BOOST_VERSION >= 104400
    void load(boost::archive::class_id_reference_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::int16_t));
        t = boost::archive::class_id_reference_type(
                boost::archive::class_id_type(std::size_t(l)));
    }
    void load(boost::archive::class_id_optional_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::int16_t));
        t = boost::archive::class_id_optional_type(
                boost::archive::class_id_type(std::size_t(l)));
    }
    void load(boost::archive::class_id_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::int16_t));
        t = boost::archive::class_id_type(std::size_t(l));
    }
    void load(boost::archive::object_id_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::uint32_t));
        t = boost::archive::object_id_type(static_cast<std::size_t>(l));
    }
    void load(boost::archive::object_reference_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::uint32_t));
        t = boost::archive::object_reference_type(
                boost::archive::object_id_type(std::size_t(l)));
    }
    void load(boost::archive::tracking_type& t) {
        bool l = false;
        this->primitive_base_t::load(l);
        t = boost::archive::tracking_type(l);
    }
    void load(boost::archive::version_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::uint32_t));
        t = boost::archive::version_type(static_cast<unsigned int>(l));
    }
    void load(boost::archive::library_version_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::uint16_t));
        t = boost::archive::library_version_type(static_cast<unsigned int>(l));
    }
    void load(boost::serialization::item_version_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::int64_t));
        if (l > static_cast<boost::int64_t>((std::numeric_limits<unsigned int>::max)())) {
            BOOST_THROW_EXCEPTION(portable_binary_iarchive_exception());
        }
        t = boost::serialization::item_version_type(static_cast<unsigned int>(l));
    }
    void load(boost::serialization::collection_size_type& t) {
        boost::int64_t l = 0;
        load_impl(l, sizeof(boost::int64_t));
        if (l > static_cast<boost::int64_t>((std::numeric_limits<unsigned int>::max)())) {
            BOOST_THROW_EXCEPTION(portable_binary_iarchive_exception());
        }
        t = boost::serialization::collection_size_type(static_cast<unsigned int>(l)); //-V106
    }
#endif
#ifndef BOOST_NO_STD_WSTRING
    void load(std::wstring& t) {
        this->primitive_base_t::load(t);
    }
#endif
    void load(float& t) {
        this->primitive_base_t::load(t);
    }
    void load(double& t) {
        this->primitive_base_t::load(t);
    }
    void load(char& t) {
        this->primitive_base_t::load(t);
    }
    void load(unsigned char& t) {
        this->primitive_base_t::load(t);
    }
    void load(signed char& t) {
        this->primitive_base_t::load(t);
    }

    // intermediate level to support override of operators
    // for templates in the absence of partial function
    // template ordering
    typedef boost::archive::detail::common_iarchive<portable_binary_iarchive>
        detail_common_iarchive;

    template <typename T>
    void load_override(T& t, BOOST_PFTO int) {
        this->detail_common_iarchive::load_override(t, 0);
    }

    HPX_ALWAYS_EXPORT void
    load_override(boost::archive::class_name_type& t, int);

    // binary files don't include the optional information
    void load_override(boost::archive::class_id_optional_type&, int) {}

    HPX_ALWAYS_EXPORT boost::uint32_t init(boost::uint32_t flags);

public:
    template <typename Container>
    portable_binary_iarchive(Container const& buffer,
            boost::uint64_t inbound_data_size, unsigned flags = 0)
      : primitive_base_t(buffer, inbound_data_size),
        archive_base_t(flags)
    {
        this->set_flags(init(flags));
    }

    template <typename Container>
    portable_binary_iarchive(Container const& buffer, std::vector<serialization_chunk>* chunks,
            boost::uint64_t inbound_data_size, unsigned flags = 0)
      : primitive_base_t(buffer, chunks, inbound_data_size),
        archive_base_t(flags)
    {
        this->set_flags(init(flags));
    }

    // the optimized load_array dispatches to load_binary
    template <typename T>
    void load_array(boost::serialization::array<T>& a, unsigned int)
    {
        // If we need to potentially flip bytes we serialize each element
        // separately.
#ifdef BOOST_BIG_ENDIAN
        if (this->flags() & (endian_little | disable_array_optimization)) {
            for (std::size_t i = 0; i != a.count(); ++i)
                load(a.address()[i]);
        }
#else
        if (this->flags() & (endian_big | disable_array_optimization)) {
            for (std::size_t i = 0; i != a.count(); ++i)
                load(a.address()[i]);
        }
#endif
        else {
            this->primitive_base_t::load_array(a);
        }
    }

    void load_array(boost::serialization::array<float>& a, unsigned int)
    {
        this->primitive_base_t::load_array(a);
    }
    void load_array(boost::serialization::array<double>& a, unsigned int)
    {
        this->primitive_base_t::load_array(a);
    }
    void load_array(boost::serialization::array<char>& a, unsigned int)
    {
        this->primitive_base_t::load_array(a);
    }
    void load_array(boost::serialization::array<unsigned char>& a, unsigned int)
    {
        this->primitive_base_t::load_array(a);
    }
    void load_array(boost::serialization::array<signed char>& a, signed int)
    {
        this->primitive_base_t::load_array(a);
    }
};

#undef HPX_SERIALIZATION_EXPORT
}}

// required by export in boost version > 1.34
BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_iarchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(hpx::util::portable_binary_iarchive)

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace archive { namespace detail
{
    template <>
    struct load_pointer_type<hpx::util::portable_binary_iarchive>
    {
        typedef hpx::util::portable_binary_iarchive archive_type;

        struct abstract
        {
            template <typename T>
            static const basic_pointer_iserializer*
            register_type(archive_type & /* ar */)
            {
                // it has? to be polymorphic
                BOOST_STATIC_ASSERT(boost::is_polymorphic< T >::value);
                return static_cast<basic_pointer_iserializer *>(NULL);
             }
        };

        struct non_abstract
        {
            template <typename T>
            static const basic_pointer_iserializer*
            register_type(archive_type& ar)
            {
                return ar.register_type(static_cast<T *>(NULL));
            }
        };

        template <typename T>
        static const basic_pointer_iserializer*
        register_type(archive_type &ar, const T & /*t*/)
        {
            // there should never be any need to load an abstract polymorphic
            // class pointer.  Inhibiting code generation for this
            // permits abstract base classes to be used - note: exception
            // virtual serialize functions used for plug-ins
            typedef BOOST_DEDUCED_TYPENAME
                mpl::eval_if<
                    boost::serialization::is_abstract<const T>,
                    boost::mpl::identity<abstract>,
                    boost::mpl::identity<non_abstract>
                >::type typex;
            return typex::template register_type< T >(ar);
        }

        template <typename T>
        static T * pointer_tweak(
            const boost::serialization::extended_type_info & eti,
            void const * const t,
            const T &)
        {
            // tweak the pointer back to the base class
            return static_cast<T *>(
                const_cast<void *>(
                    boost::serialization::void_upcast(
                        eti,
                        boost::serialization::singleton<
                            BOOST_DEDUCED_TYPENAME
                            boost::serialization::type_info_implementation< T >::type
                        >::get_const_instance(),
                        t
                    )
                )
            );
        }

        template <typename T>
        static void check_load(T& /* t */)
        {
            check_pointer_level< T >();
            // check_pointer_tracking< T >();      // this has to be disabled to avoid warnings
        }

        static const basic_pointer_iserializer *
        find(const boost::serialization::extended_type_info & type)
        {
            return static_cast<const basic_pointer_iserializer *>(
                archive_serializer_map<archive_type>::find(type)
            );
        }

        template <typename Tptr>
        static void invoke(archive_type & ar, Tptr & t)
        {
            check_load(*t);
            const basic_pointer_iserializer * bpis_ptr = register_type(ar, *t);
            const basic_pointer_iserializer * newbpis_ptr = ar.load_pointer(
                // note major hack here !!!
                // I tried every way to convert Tptr &t (where Tptr might
                // include const) to void * &.  This is the only way
                // I could make it work. RR
                (void * & )t,
                bpis_ptr,
                find
            );
            // if the pointer isn't that of the base class
            if (newbpis_ptr != bpis_ptr){
                t = pointer_tweak(newbpis_ptr->get_eti(), t, *t);
            }
        }
    };
}}}

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

#endif // HPX_USE_PORTABLE_ARCHIVES == 0
#endif // PORTABLE_BINARY_IARCHIVE_HPP
