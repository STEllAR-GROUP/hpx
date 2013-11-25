#ifndef PORTABLE_BINARY_OARCHIVE_HPP
#define PORTABLE_BINARY_OARCHIVE_HPP

#include <boost/version.hpp>
#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_integral.hpp>

#if !defined(HPX_USE_PORTABLE_ARCHIVES) || HPX_USE_PORTABLE_ARCHIVES == 0
#include <boost/archive/binary_oarchive.hpp>

namespace hpx { namespace util
{
    typedef boost::archive::binary_oarchive portable_binary_oarchive;
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
// portable_binary_oarchive.hpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
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
#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#if BOOST_VERSION >= 104400
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/collection_size_type.hpp>
#endif

#include <hpx/config.hpp>
#include <hpx/util/portable_binary_archive.hpp>
#include <hpx/util/basic_binary_oprimitive.hpp>

namespace hpx { namespace util
{

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// exception to be thrown if integer read from archive doesn't fit
// variable being loaded
class HPX_ALWAYS_EXPORT portable_binary_oarchive_exception :
    public virtual boost::archive::archive_exception
{
public:
    enum exception_code {
        invalid_flags
    };
    portable_binary_oarchive_exception(exception_code c = invalid_flags)
      : boost::archive::archive_exception(
          static_cast<boost::archive::archive_exception::exception_code>(c))
    {}
    virtual const char *what() const throw()
    {
        const char *msg = "programmer error";
        switch (static_cast<exception_code>(code)) {
        case invalid_flags:
            msg = "cannot be both big and little endian";
        default:
            boost::archive::archive_exception::what();
        }
        return msg;
    }
};

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// "Portable" output binary archive.  This is a variation of the native binary
// archive. it addresses integer size and endianness so that binary archives can
// be passed across systems. Note:floating point types are passed through as is.

#if defined(BOOST_MSVC) || defined(BOOST_INTEL_WIN)
#define HPX_SERIALIZATION_EXPORT
#else
#define HPX_SERIALIZATION_EXPORT HPX_ALWAYS_EXPORT
#endif

class HPX_SERIALIZATION_EXPORT portable_binary_oarchive :
    public hpx::util::basic_binary_oprimitive<
        portable_binary_oarchive
    >,
    public boost::archive::detail::common_oarchive<
        portable_binary_oarchive
    >
{
    typedef hpx::util::basic_binary_oprimitive<
        portable_binary_oarchive
    > primitive_base_t;
    typedef boost::archive::detail::common_oarchive<
        portable_binary_oarchive
    > archive_base_t;
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
    friend archive_base_t;
    friend primitive_base_t; // since with override save below
    friend class boost::archive::detail::interface_oarchive<
        portable_binary_oarchive
    >;
    friend class boost::archive::save_access;
protected:
#endif

    HPX_ALWAYS_EXPORT void save_impl(boost::int64_t const l, char const maxsize);
    HPX_ALWAYS_EXPORT void save_impl(boost::uint64_t const l, char const maxsize);

    // default fall through for any types not specified here
    template <typename T>
    void save_integral(T const& val, boost::mpl::false_)
    {
        save_impl(static_cast<boost::int64_t>(val), sizeof(T));
    }

    template <typename T>
    void save_integral(T const& val, boost::mpl::true_)
    {
        save_impl(static_cast<boost::uint64_t>(val), sizeof(T));
    }

    template <typename T>
    void save(T const& t, typename boost::enable_if<boost::is_integral<T> >::type* = 0)
    {
        save_integral(t, typename boost::is_unsigned<T>::type());
    }

    template <typename T>
    void save(T const& t, typename boost::disable_if<boost::is_integral<T> >::type* = 0)
    {
        this->primitive_base_t::save(t);
    }

    void save(std::string const &t)
    {
        this->primitive_base_t::save(t);
    }

#if BOOST_VERSION >= 104400
    void save(const boost::archive::class_id_reference_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::class_id_optional_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::class_id_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::int16_t));
    }
    void save(const boost::archive::object_id_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::object_reference_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::tracking_type& t) {
        bool l = t;
        this->primitive_base_t::save(l);
    }
    void save(const boost::archive::version_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::uint32_t));
    }
    void save(const boost::archive::library_version_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::uint16_t));
    }
    void save(const boost::serialization::item_version_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::int64_t));
    }
    void save(const boost::serialization::collection_size_type& t) {
        boost::int64_t l = t;
        save_impl(l, sizeof(boost::int64_t));
    }
#endif
#ifndef BOOST_NO_STD_WSTRING
    void save(std::wstring const& t) {
        this->primitive_base_t::save(t);
    }
#endif
    void save(float const& t) {
        this->primitive_base_t::save(t);
    }
    void save(double const& t) {
        this->primitive_base_t::save(t);
    }
    void save(char const& t) {
        this->primitive_base_t::save(t);
    }
    void save(unsigned char const& t) {
        this->primitive_base_t::save(t);
    }
    void save(signed char const& t) {
        this->primitive_base_t::save(t);
    }

    // default processing - kick back to base class.  Note the
    // extra stuff to get it passed borland compilers
    typedef boost::archive::detail::common_oarchive<portable_binary_oarchive>
        detail_common_oarchive;

    template <typename T>
    void save_override(T & t, BOOST_PFTO int) {
        this->detail_common_oarchive::save_override(t, 0);
    }
    // explicitly convert to char * to avoid compile ambiguities
    void save_override(const boost::archive::class_name_type & t, int) {
        const std::string s(t);
        *this << s;
    }

    // binary files don't include the optional information
    void save_override(const boost::archive::class_id_optional_type&, int) {}

    HPX_ALWAYS_EXPORT void init(util::binary_filter* filter, unsigned int flags);

public:
    template <typename Container>
    portable_binary_oarchive(Container& buffer, binary_filter* filter = 0, unsigned flags = 0)
      : primitive_base_t(buffer, flags),
        archive_base_t(flags)
    {
        init(filter, flags);
    }

    template <typename Container>
    portable_binary_oarchive(Container& buffer, std::vector<serialization_chunk>* chunks,
            binary_filter* filter = 0, unsigned flags = 0)
      : primitive_base_t(buffer, chunks, flags),
        archive_base_t(flags)
    {
        init(filter, flags);
    }

    // the optimized save_array dispatches to the base class save_array
    // implementation

    // default fall through for any types not specified here
    template <typename T>
    void save_array(boost::serialization::array<T> const& a, unsigned int)
    {
        // If we need to potentially flip bytes we serialize each element
        // separately.
#ifdef BOOST_BIG_ENDIAN
        if (this->flags() & (endian_little | disable_array_optimization)) {
            for (std::size_t i = 0; i != a.count(); ++i)
                save(a.address()[i]);
        }
#else
        if (this->flags() & (endian_big | disable_array_optimization)) {
            for (std::size_t i = 0; i != a.count(); ++i)
                save(a.address()[i]);
        }
#endif
        else {
            this->primitive_base_t::save_array(a);
        }
    }

    void save_array(boost::serialization::array<float> const& a, unsigned int)
    {
        this->primitive_base_t::save_array(a);
    }
    void save_array(boost::serialization::array<double> const& a, unsigned int)
    {
        this->primitive_base_t::save_array(a);
    }
    void save_array(boost::serialization::array<char> const& a, unsigned int)
    {
        this->primitive_base_t::save_array(a);
    }
    void save_array(boost::serialization::array<unsigned char> const& a, unsigned int)
    {
        this->primitive_base_t::save_array(a);
    }
    void save_array(boost::serialization::array<signed char> const& a, unsigned int)
    {
        this->primitive_base_t::save_array(a);
    }
};

#undef HPX_SERIALIZATION_EXPORT
}}

// required by export in boost version > 1.34
BOOST_SERIALIZATION_REGISTER_ARCHIVE(hpx::util::portable_binary_oarchive)
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(hpx::util::portable_binary_oarchive)

///////////////////////////////////////////////////////////////////////////////
// We provide a specialization for the oserializer for pointer types to
// disable the warning produced by Boost.Serialization when pointers are
// serialized with tracking disabled.

#include <boost/archive/detail/oserializer.hpp>

namespace boost { namespace archive { namespace detail
{
    template <>
    struct save_non_pointer_type<hpx::util::portable_binary_oarchive>
    {
        typedef hpx::util::portable_binary_oarchive archive_type;

        // note this bounces the call right back to the archive
        // with no runtime overhead
        struct save_primitive
        {
            template <typename T>
            static void invoke(archive_type & ar, const T & t)
            {
                save_access::save_primitive(ar, t);
            }
        };

        // same as above but passes through serialization
        struct save_only
        {
            template <typename T>
            static void invoke(archive_type & ar, const T & t)
            {
                // make sure call is routed through the highest interface that might
                // be specialized by the user.
                boost::serialization::serialize_adl(
                    ar,
                    const_cast<T &>(t),
                    ::boost::serialization::version< T >::value
                );
            }
        };

        // adds class information to the archive. This includes
        // serialization level and class version
        struct save_standard
        {
            template <typename T>
            static void invoke(archive_type &ar, const T & t)
            {
                ar.save_object(
                    & t,
                    boost::serialization::singleton<
                        oserializer<archive_type, T>
                    >::get_const_instance()
                );
            }
        };

        // adds class information to the archive. This includes
        // serialization level and class version
        struct save_conditional
        {
            template <typename T>
            static void invoke(archive_type &ar, const T &t)
            {
                //if(0 == (ar.get_flags() & no_tracking))
                    save_standard::invoke(ar, t);
                //else
                //   save_only::invoke(ar, t);
            }
        };


        template <typename T>
        static void invoke(archive_type & ar, const T & t)
        {
            typedef
                BOOST_DEDUCED_TYPENAME mpl::eval_if<
                // if its primitive
                    mpl::equal_to<
                        boost::serialization::implementation_level< T >,
                        mpl::int_<boost::serialization::primitive_type>
                    >,
                    mpl::identity<save_primitive>,
                // else
                BOOST_DEDUCED_TYPENAME mpl::eval_if<
                    // class info / version
                    mpl::greater_equal<
                        boost::serialization::implementation_level< T >,
                        mpl::int_<boost::serialization::object_class_info>
                    >,
                    // do standard save
                    mpl::identity<save_standard>,
                // else
                BOOST_DEDUCED_TYPENAME mpl::eval_if<
                        // no tracking
                    mpl::equal_to<
                        boost::serialization::tracking_level< T >,
                        mpl::int_<boost::serialization::track_never>
                    >,
                    // do a fast save
                    mpl::identity<save_only>,
                // else
                    // do a fast save only tracking is turned off
                    mpl::identity<save_conditional>
                > > >::type typex;
            check_object_versioning< T >();
            typex::invoke(ar, t);
        }

        template <typename T>
        static void invoke(archive_type & ar, T & t)
        {
            check_object_level< T >();
            //check_object_tracking< T >();      // this has to be disabled to avoid warnings
            invoke(ar, const_cast<const T &>(t));
        }
    };

    template <>
    struct save_pointer_type<hpx::util::portable_binary_oarchive>
    {
        typedef hpx::util::portable_binary_oarchive archive_type;

        struct abstract
        {
            template <typename T>
            static const basic_pointer_oserializer*
            register_type(archive_type& /* ar */)
            {
                // it has? to be polymorphic
                BOOST_STATIC_ASSERT(boost::is_polymorphic<T>::value);
                return NULL;
            }
        };

        struct non_abstract
        {
            template <typename T>
            static const basic_pointer_oserializer*
            register_type(archive_type& ar)
            {
                return ar.register_type(static_cast<T*>(NULL));
            }
        };

        template <typename T>
        static const basic_pointer_oserializer*
        register_type(archive_type &ar, T& /*t*/)
        {
            // there should never be any need to save an abstract polymorphic
            // class pointer.  Inhibiting code generation for this
            // permits abstract base classes to be used - note: exception
            // virtual serialize functions used for plug-ins
            typedef
                BOOST_DEDUCED_TYPENAME mpl::eval_if<
                    boost::serialization::is_abstract<T>,
                    mpl::identity<abstract>,
                    mpl::identity<non_abstract>
                >::type typex;
            return typex::template register_type<T>(ar);
        }

        struct non_polymorphic
        {
            template <typename T>
            static void save(archive_type &ar, T& t)
            {
                const basic_pointer_oserializer & bpos =
                    boost::serialization::singleton<
                        pointer_oserializer<archive_type, T>
                    >::get_const_instance();
                // save the requested pointer type
                ar.save_pointer(&t, &bpos);
            }
        };

        struct polymorphic
        {
            template <typename T>
            static void save(archive_type &ar,T& t)
            {
                BOOST_DEDUCED_TYPENAME
                boost::serialization::type_info_implementation<T>::type const
                & i = boost::serialization::singleton<
                    BOOST_DEDUCED_TYPENAME
                    boost::serialization::type_info_implementation<T>::type
                >::get_const_instance();

                boost::serialization::extended_type_info const * const this_type = & i;

                // retrieve the true type of the object pointed to
                // if this assertion fails its an error in this library
                HPX_ASSERT(NULL != this_type);

                const boost::serialization::extended_type_info * true_type =
                    i.get_derived_extended_type_info(t);

                // note:if this exception is thrown, be sure that derived pointer
                // is either registered or exported.
                if (NULL == true_type) {
                    boost::serialization::throw_exception(
                        archive_exception(
                            archive_exception::unregistered_class,
                            "derived class not registered or exported"
                        )
                    );
                }

                // if its not a pointer to a more derived type
                const void *vp = static_cast<const void *>(&t);
                if(*this_type == *true_type){
                    const basic_pointer_oserializer * bpos = register_type(ar, t);
                    ar.save_pointer(vp, bpos);
                    return;
                }
                // convert pointer to more derived type. if this is thrown
                // it means that the base/derived relationship hasn't be registered
                vp = serialization::void_downcast(
                    *true_type,
                    *this_type,
                    static_cast<const void *>(&t)
                );
                if(NULL == vp){
                    boost::serialization::throw_exception(
                        archive_exception(
                            archive_exception::unregistered_cast,
                            true_type->get_debug_info(),
                            this_type->get_debug_info()
                        )
                    );
                }

                // since true_type is valid, and this only gets made if the
                // pointer oserializer object has been created, this should never
                // fail
                const basic_pointer_oserializer * bpos
                    = static_cast<const basic_pointer_oserializer *>(
                        boost::serialization::singleton<
                            archive_serializer_map<archive_type>
                        >::get_const_instance().find(*true_type)
                    );
                HPX_ASSERT(NULL != bpos);
                if(NULL == bpos)
                    boost::serialization::throw_exception(
                        archive_exception(
                            archive_exception::unregistered_class,
                            "derived class not registered or exported"
                        )
                    );
                ar.save_pointer(vp, bpos);
            }
        };

        template <typename T>
        static void save(archive_type& ar, const T& t)
        {
            check_pointer_level<T>();
            //check_pointer_tracking<T>();      // this has to be disabled to avoid warnings
            typedef BOOST_DEDUCED_TYPENAME mpl::eval_if<
                is_polymorphic<T>,
                mpl::identity<polymorphic>,
                mpl::identity<non_polymorphic>
            >::type type;
            type::save(ar, const_cast<T &>(t));
        }

        template <typename TPtr>
        static void invoke(archive_type& ar, const TPtr t)
        {
            register_type(ar, *t);
            if (NULL == t){
                basic_oarchive & boa
                    = boost::serialization::smart_cast_reference<basic_oarchive &>(ar);
                boa.save_null_pointer();
                save_access::end_preamble(ar);
                return;
            }
            save(ar, *t);
        }
    };
}}}

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

#endif // HPX_USE_PORTABLE_ARCHIVES == 0
#endif // PORTABLE_BINARY_OARCHIVE_HPP


