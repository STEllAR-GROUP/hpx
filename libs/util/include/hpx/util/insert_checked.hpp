////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CA606CBC_A6AA_44A8_BDB0_46A3A08BAA82)
#define HPX_CA606CBC_A6AA_44A8_BDB0_46A3A08BAA82

#include <utility>

namespace hpx { namespace util
{

////////////////////////////////////////////////////////////////////////////////
/// \brief  Helper function for writing predicates that test whether an std::map
///         insertion succeeded. This inline template function negates the need
///         to explicitly write the sometimes lengthy std::pair<Iterator, bool>
///         type.
///
/// \param r  [in] The return value of a std::map insert operation.
///
/// \returns  This function returns \b r.second.
template <typename Iterator>
inline bool insert_checked(std::pair<Iterator, bool> const& r)
{
    return r.second;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief  Helper function for writing predicates that test whether an std::map
///         insertion succeeded. This inline template function negates the need
///         to explicitly write the sometimes lengthy std::pair<Iterator, bool>
///         type.
///
/// \param r  [in] The return value of a std::map insert operation.
///
/// \param r  [out] A reference to an Iterator, which is set to \b r.first.
///
/// \returns  This function returns \b r.second.
template <typename Iterator>
inline bool insert_checked(std::pair<Iterator, bool> const& r, Iterator& it)
{
    it = r.first;
    return r.second;
}

}}

#endif // HPX_CA606CBC_A6AA_44A8_BDB0_46A3A08BAA82

