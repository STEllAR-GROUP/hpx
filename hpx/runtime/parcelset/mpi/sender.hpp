//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_MPI_SENDER_HPP
#define HPX_PARCELSET_MPI_SENDER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/mpi/header.hpp>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/noncopyable.hpp>

#include <vector>
#include <utility>

namespace hpx { namespace parcelset { namespace mpi {

    struct sender
      : boost::noncopyable
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        enum sender_state
        {
            invalid,
            sending_header,
            sent_header,
            sending_data,
            sent_data,
            sender_done,
        };

        sender(
            header const & h,
            BOOST_RV_REF(std::vector<char>) buffer,
            BOOST_RV_REF(std::vector<write_handler_type>) handlers,
            MPI_Comm communicator)
          : communicator_(communicator)
          , header_(h)
          , buffer_(boost::move(buffer))
          , handlers_(boost::move(handlers))
          , state_(invalid)
        {
            header_.assert_valid();
            BOOST_ASSERT(header_.rank() != util::mpi_environment::rank());

            MPI_Irsend(
                header_.data(), // Data pointer
                2,              // Size
                header_.type(), // MPI Datatype
                header_.rank(), // Destination
                0,              // Tag
                communicator_,  // Communicator
                &header_request_       // Request
                );
            state_ = sending_header;
        }

        bool done()
        {
            switch (state_)
            {
                case sending_header:
                    {
                        int completed = 0;
                        MPI_Test(&header_request_, &completed, MPI_STATUS_IGNORE);
                        if(completed)
                        {
                            state_ = sent_header;
                            return done();
                        }
                        break;
                    }
                case sent_header:
                    {
                        BOOST_ASSERT(static_cast<std::size_t>(header_.size()) == buffer_.size());
                        MPI_Irsend(
                            buffer_.data(), // Data pointer
                            static_cast<int>(buffer_.size()), // Size
                            MPI_CHAR,       // MPI Datatype
                            header_.rank(), // Destination
                            header_.tag(),  // Tag
                            communicator_,  // Communicator
                            &data_request_       // Request
                            );
                        state_ = sending_data;
                        return done();
                    }
                case sending_data:
                    {
                        int completed = 0;
                        MPI_Test(&data_request_, &completed, MPI_STATUS_IGNORE);
                        if(completed)
                        {
                            state_ = sent_data;
                            return done();
                        }
                        break;
                    }
                case sent_data:
                    {
                        error_code ec;
                        BOOST_FOREACH(write_handler_type & f, handlers_)
                        {
                            if(!f.empty())
                            {
                                f(ec, header_.size());
                            }
                        }
                        state_ = sender_done;
                        return done();
                    }
                case sender_done:
                    return true;
                default:
                case invalid:
                    {
                        BOOST_ASSERT(false);
                    }
                    return false;
            }
            return false;
        }

        int tag() const
        {
            return header_.tag();
        }

        private:
            MPI_Comm communicator_;
            header header_;
            std::vector<char> buffer_;
            std::vector<write_handler_type> handlers_;
            sender_state state_;

            MPI_Request header_request_;
            MPI_Request data_request_;
    };

}}}

#endif
