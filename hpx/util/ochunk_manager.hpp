//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_OCHUNK_MANAGER_JUL_31_2013_1158AM)
#define HPX_UTIL_OCHUNK_MANAGER_JUL_31_2013_1158AM

#include <hpx/util/binary_filter.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    struct erase_ocontainer_type
    {
        virtual ~erase_ocontainer_type() {}
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void save_binary(void const* address, std::size_t count) = 0;
        virtual void save_binary_chunk(void const* address, std::size_t count) = 0;
    };

    template <typename Container>
    struct ocontainer_type : erase_ocontainer_type
    {
    private:
        std::size_t get_chunk_size(std::size_t chunk) const
        {
            return (*chunks_)[chunk].size_;
        }

        void set_chunk_size(std::size_t chunk, std::size_t size)
        {
            (*chunks_)[chunk].size_ = size;
        }

        chunk_type get_chunk_type(std::size_t chunk) const
        {
            return (*chunks_)[chunk].type_;
        }

        chunk_data get_chunk_data(std::size_t chunk) const
        {
            return (*chunks_)[chunk].data_;
        }

        std::size_t get_num_chunks() const
        {
            return chunks_->size();
        }

    public:
        ocontainer_type(Container& cont)
          : cont_(cont), current_(0), start_compressing_at_(0), filter_(0),
            chunks_(0), current_chunk_(std::size_t(-1))
        {}

        ocontainer_type(Container& cont, std::vector<serialization_chunk>* chunks)
          : cont_(cont), current_(0), start_compressing_at_(0), filter_(0),
            chunks_(chunks), current_chunk_(std::size_t(-1))
        {
            if (chunks_)
            {
                chunks_->clear();
                chunks_->push_back(create_index_chunk(0, 0));
                current_chunk_ = 0;
            }
        }

        ~ocontainer_type()
        {
            if (filter_) {
                std::size_t written = 0;

                if (cont_.size() < current_)
                    cont_.resize(current_);
                current_ = start_compressing_at_;

                do {
                    bool flushed = filter_->flush(&cont_[current_],
                        cont_.size()-current_, written);

                    current_ += written;
                    if (flushed)
                        break;

                    // resize container
                    cont_.resize(cont_.size()*2);

                } while (true);

                cont_.resize(current_);         // truncate container
            }
            else if (chunks_) {
                BOOST_ASSERT(
                    get_chunk_type(current_chunk_) == chunk_type_index ||
                    get_chunk_size(current_chunk_) != 0);

                // complement current serialization_chunk by setting its length
                if (get_chunk_type(current_chunk_) == chunk_type_index)
                {
                    BOOST_ASSERT(get_chunk_size(current_chunk_) == 0);

                    set_chunk_size(current_chunk_,
                        current_ - get_chunk_data(current_chunk_).index_);
                }
            }
        }

        void set_filter(binary_filter* filter)
        {
            BOOST_ASSERT(0 == filter_);
            filter_ = filter;
            start_compressing_at_ = current_;

            if (chunks_) {
                BOOST_ASSERT(get_num_chunks() == 1 && get_chunk_size(0) == 0);
                chunks_->clear();
            }
        }

        void save_binary(void const* address, std::size_t count)
        {
            BOOST_ASSERT(count != 0);
            {
                if (filter_) {
                    filter_->save(address, count);
                }
                else {
                    // make sure there is a current serialization_chunk descriptor available
                    if (chunks_ && (get_chunk_type(current_chunk_) == chunk_type_pointer ||
                        get_chunk_size(current_chunk_) != 0))
                    {
                        // add a new serialization_chunk
                        chunks_->push_back(create_index_chunk(current_, 0));
                        ++current_chunk_;
                    }

                    if (cont_.size() < current_ + count)
                        cont_.resize(cont_.size() + count);

                    if (count == 1)
                        cont_[current_] = *static_cast<unsigned char const*>(address);
                    else
                        std::memcpy(&cont_[current_], address, count);
                }
                current_ += count;
            }
        }

        void save_binary_chunk(void const* address, std::size_t count)
        {
            if (filter_ || chunks_ == 0 || count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD) {
                // fall back to serialization_chunk-less archive
                this->ocontainer_type::save_binary(address, count);
            }
            else {
                BOOST_ASSERT(
                    get_chunk_type(current_chunk_) == chunk_type_index ||
                    get_chunk_size(current_chunk_) != 0);

                // complement current serialization_chunk by setting its length
                if (get_chunk_type(current_chunk_) == chunk_type_index)
                {
                    BOOST_ASSERT(get_chunk_size(current_chunk_) == 0);

                    set_chunk_size(current_chunk_,
                        current_ - get_chunk_data(current_chunk_).index_);
                }

                // add a new serialization_chunk referring to the external buffer
                chunks_->push_back(create_pointer_chunk(address, count));
                ++current_chunk_;
            }
        }

        Container& cont_;
        std::size_t current_;
        std::size_t start_compressing_at_;
        binary_filter* filter_;

        std::vector<serialization_chunk>* chunks_;
        std::size_t current_chunk_;
    };
}}}

#endif

