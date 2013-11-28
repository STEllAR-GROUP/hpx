//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Parts of this code were taken from the Boost.IoStreams library
//  Copyright (c) 2004 Jonathan Turkanis
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_53514563_F7A4_4F1E_B52A_083D8033E014)
#define HPX_53514563_F7A4_4F1E_B52A_083D8033E014

#include <algorithm>                       // copy
#include <iosfwd>                          // streamsize

#include <hpx/util/assert.hpp>
#include <boost/iostreams/categories.hpp>  // sink_tag

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{

/// This is a Boost.IoStreams Sink that can be used to create an [io]stream
/// on top of any class that fulfills STL OutputIterator.
template <typename Iterator, typename Char = char>
struct iterator_sink
{
    // The STL OutputIterator concept does not require that an OutputIterator
    // define a nested value_type.
    typedef Char char_type;
    typedef boost::iostreams::sink_tag category;

    explicit iterator_sink(Iterator const& it) : it_(it) {}

    template <typename Container>
    iterator_sink(Container& c) : it_(Iterator(c)) {}

    // Write up to n characters to the underlying data sink into the
    // buffer s, returning the number of characters written.
    std::streamsize write(const char_type* s, std::streamsize n)
    {
        std::copy(s, s + n, it_);
        return n;
    }

    Iterator& iterator() { return it_; }

  private:
    Iterator it_;
};

///////////////////////////////////////////////////////////////////////////////
}}

#endif // HPX_53514563_F7A4_4F1E_B52A_083D8033E014

