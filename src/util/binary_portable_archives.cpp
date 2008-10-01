//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// export the defined functions
#define BOOST_ARCHIVE_SOURCE

// this hack is needed to properly compiler this shared library, allowing to 
// export the symbols and auto link with the serialization 
#if !defined(BOOST_ALL_NO_LIB) && !defined(BOOST_SERIALIZATION_NO_LIB)
// Set the name of our library, this will get undef'ed by auto_link.hpp
// once it's done with it:
#define BOOST_LIB_NAME boost_serialization

// If we're importing code from a dll, then tell auto_link.hpp about it:
#if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_SERIALIZATION_DYN_LINK)
#  define BOOST_DYN_LINK
#endif

// And include the header that does the work:
#include <boost/config/auto_link.hpp>
#endif  // auto-linking disabled

#include <hpx/hpx_fwd.hpp>

#include <boost/serialization/serialization.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

///////////////////////////////////////////////////////////////////////////////
// explicit template instantiations for our portable archives

// required by export in boost <= 1.34
#define BOOST_ARCHIVE_CUSTOM_OARCHIVE_TYPES hpx::util::portable_binary_oarchive

// explicitly instantiate for this type of text stream
#include <boost/archive/impl/basic_binary_oarchive.ipp>
#include <boost/archive/impl/archive_pointer_oserializer.ipp>
#include <boost/archive/impl/basic_binary_oprimitive.ipp>

namespace boost { namespace archive 
{
    // explicitly instantiate for this type of binary stream
    template class binary_oarchive_impl<
        hpx::util::portable_binary_oarchive, 
        std::ostream::char_type, 
        std::ostream::traits_type
    >;
    template class detail::archive_pointer_oserializer<
        hpx::util::portable_binary_oarchive>;

}} // namespace boost::archive

///////////////////////////////////////////////////////////////////////////////
// same for input archive

// required by export in boost <= 1.34
#define BOOST_ARCHIVE_CUSTOM_IARCHIVE_TYPES hpx::util::portable_binary_iarchive

// explicitly instantiate for this type of text stream
#include <boost/archive/impl/basic_binary_iarchive.ipp>
#include <boost/archive/impl/archive_pointer_iserializer.ipp>
#include <boost/archive/impl/basic_binary_iprimitive.ipp>

namespace boost { namespace archive 
{
    template class binary_iarchive_impl<
        hpx::util::portable_binary_iarchive, 
        std::istream::char_type, 
        std::istream::traits_type
    >;
    template class detail::archive_pointer_iserializer<
        hpx::util::portable_binary_iarchive>;

}} // namespace boost::archive


