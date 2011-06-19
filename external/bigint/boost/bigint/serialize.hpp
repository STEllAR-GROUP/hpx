/* Boost bigint.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_SERIALIZE_HPP
#define BOOST_BIGINT_BIGINT_SERIALIZE_HPP

#include <boost/bigint/bigint.hpp>

#ifndef BOOST_SERIALIZATION_DEFAULT_TYPE_INFO
#include <boost/serialization/extended_type_info_typeid.hpp>
#endif

#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization {

template <class Archive, typename I> void save(Archive & ar, const bigint_base<I> & t, const unsigned int)
{
	std::string data = t.str();
	ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive, typename I> void load(Archive & ar, bigint_base<I> & t, const unsigned int)
{
    std::string data;
    ar & BOOST_SERIALIZATION_NVP(data);

    t = bigint_base<I>(data); // load_construct_data ?
}

template <class Archive, typename I> inline void serialize(Archive & ar, bigint_base<I> & t, const unsigned int file_version)
{
    split_free(ar, t, file_version); 
}

} } // namespace boost::serialization

#endif // BOOST_BIGINT_BIGINT_SERIALIZE_HPP
