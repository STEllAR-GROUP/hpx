//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.IoStreams library
//  Copyright (c) 2004 Jonathan Turkanis
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CONTAINER_DEVICE_MAR_24_2008_0955AM)
#define HPX_UTIL_CONTAINER_DEVICE_MAR_24_2008_0955AM

#include <algorithm>                       // copy, min
#include <iosfwd>                          // streamsize

#include <boost/assert.hpp>
#include <boost/iostreams/categories.hpp>  // source_tag
#include <boost/iostreams/positioning.hpp> // stream_offset

#include <hpx/exception.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    /// this is a Boost.IoStreams SeekableDevice that can be used to create
    /// an [io]stream on top of a random access container (i.e. vector<>)
    template<typename Container>
    class container_device
    {
    public:
        typedef typename Container::value_type char_type;
        typedef boost::iostreams::seekable_device_tag category;

        container_device(Container& container)
          : container_(container), pos_(0)
        {}

        /// Read up to n characters from the underlying data source into the
        /// buffer s, returning the number of characters read; return -1 to
        /// indicate EOF
        std::streamsize read(char_type* s, std::streamsize n)
        {
            std::streamsize amt =
                static_cast<std::streamsize>(container_.size() - pos_);
            std::streamsize result = (std::min)(n, amt);
            if (result != 0) {
                std::copy(container_.begin() + pos_,
                          container_.begin() + pos_ + result, s);
                pos_ += result;
                return result;
            }
            else {
                return -1;  // EOF
            }
        }

        /// Write up to n characters to the underlying data sink into the
        /// buffer s, returning the number of characters written
        std::streamsize write(const char_type* s, std::streamsize n)
        {
            std::streamsize result = 0;
            if (pos_ != container_.size()) {
                std::streamsize amt =
                    static_cast<std::streamsize>(container_.size() - pos_);
                std::streamsize result = (std::min)(n, amt);
                std::copy(s, s + result, container_.begin() + pos_);
                pos_ += result;
            }
            if (result < n) {
                container_.insert(container_.end(), s, s + n);
                pos_ = container_.size();
            }
            return n;
        }

        /// Seek to position off and return the new stream position. The
        /// argument 'way' indicates how off is interpreted:
        ///    - std::ios_base::beg indicates an offset from the sequence
        ///      beginning
        ///    - std::ios_base::cur indicates an offset from the current
        ///      character position
        ///    - std::ios_base::end indicates an offset from the sequence end
        boost::iostreams::stream_offset seek(
            boost::iostreams::stream_offset off, std::ios_base::seekdir way)
        {
            // Determine new value of pos_
            boost::iostreams::stream_offset next(0);
            if (way == std::ios_base::beg) {
                next = off;
            }
            else if (way == std::ios_base::cur) {
                next = pos_ + off;
            }
            else if (way == std::ios_base::end) {
                next = container_.size() + off - 1;
            }
            else {
                BOOST_ASSERT(false);
            }

            // Check for errors
            if (next < ((boost::iostreams::stream_offset)0)
             || next >= ((boost::iostreams::stream_offset)container_.size()))
                throw std::ios_base::failure("bad seek offset");

            pos_ = (size_type)next;
            return pos_;
        }

        Container& container() { return container_; }

    private:
        typedef typename Container::size_type size_type;
        Container& container_;
        size_type pos_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif

