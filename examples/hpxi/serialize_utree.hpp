////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8B48BD01_B018_4A56_B48D_6011533A7C58)
#define HPX_8B48BD01_B018_4A56_B48D_6011533A7C58

#include <prana/include/utree.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void save(Archive& ar, boost::spirit::utree& ut, unsigned int);

template <typename Archive>
void load(Archive& ar, boost::spirit::utree& ut, unsigned int);

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::spirit::utree);

#endif // HPX_8B48BD01_B018_4A56_B48D_6011533A7C58

