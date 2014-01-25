//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/mpi/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/mpi/sender.hpp>
#include <hpx/runtime/parcelset/policies/mpi/receiver.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines += 
#if defined(HPX_PARCELPORT_MPI_ENV)
            "env = ${HPX_PARCELPORT_MPI_ENV:" HPX_PARCELPORT_MPI_ENV "}",
#else
            "env = ${HPX_PARCELPORT_MPI_ENV:PMI_RANK,OMPI_COMM_WORLD_SIZE}",
#endif
            "multithreaded = ${HPX_PARCELPORT_MPI_MULTITHREADED:0}",
            "io_pool_size = 1"
            ;

        return lines;
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , stopped_(false)
      , handling_messages_(false)
      , next_tag_(1)
    {
        if (here_.get_type() != connection_mpi) {
            HPX_THROW_EXCEPTION(network_error, "mpi::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
    }
    
    connection_handler::~connection_handler()
    {
    }

    bool connection_handler::run()
    {
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);
        acceptor_.run(communicator_);
        do_background_work();      // schedule message handler
        return true;
    }

    void connection_handler::stop()
    {
        // Mark stopped state
        stopped_ = true;
    }

    // Make sure all pending requests are handled
    void connection_handler::do_background_work()
    {
        if (stopped_)
            return;

        // Atomically set handling_messages_ to true, if another work item hasn't
        // started executing before us.
        bool false_ = false;
        if (!handling_messages_.compare_exchange_strong(false_, true))
            return;

        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        io_service.post(HPX_STD_BIND(&connection_handler::handle_messages, this));
    }
            
    std::string connection_handler::get_locality_name() const
    {
        return util::mpi_environment::get_processor_name();
    }
    
    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::shared_ptr<sender> sender_connection(new sender(
            communicator_, get_next_tag(), tag_mtx_, free_tags_, l,
            *this, this->parcels_sent_));
        
        return sender_connection;
    }

    void connection_handler::add_sender(boost::shared_ptr<sender> sender_connection)
    {
        hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
        senders_.push_back(sender_connection);
    }
    
    void add_sender(connection_handler & handler, boost::shared_ptr<sender> sender_connection)
    {
        handler.add_sender(sender_connection);
    }
            
    int connection_handler::get_next_tag()
    {
        int tag = 0;
        {
            hpx::lcos::local::spinlock::scoped_lock l(tag_mtx_);
            if(!free_tags_.empty())
            {
                tag = free_tags_.front();
                free_tags_.pop_front();
                return tag;
            }
        }
        if(next_tag_ + 1 > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            HPX_THROW_EXCEPTION(network_error, "mpi::connection_handler::get_next_tag",
                "there are no free tags available. Consider decreasing the cache size");
            return tag;
        }
        tag = static_cast<int>(next_tag_++);
        return tag;
    }
    
    namespace detail
    {
        struct handling_messages
        {
            handling_messages(boost::atomic<bool>& handling_messages_flag)
              : handling_messages_(handling_messages_flag)
            {}

            ~handling_messages()
            {
                handling_messages_.store(false);
            }

            boost::atomic<bool>& handling_messages_;
        };
    }

    void connection_handler::handle_messages()
    {
        detail::handling_messages hm(handling_messages_);       // reset on exit

        bool bootstrapping = hpx::is_starting();
        bool has_work = true;

        hpx::util::high_resolution_timer t;

        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work) || (t.elapsed() < 2.0))
        {
            // handle all send requests
            {
                hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
                for(senders_type::iterator it = senders_.begin(); it != senders_.end(); /**/)
                {
                    if((*it)->done())
                    {
                        it = senders_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                has_work = !senders_.empty();
            }
            
            // handle all receive requests
            for(receivers_type::iterator it = receivers_.begin(); it != receivers_.end(); /**/)
            {
                if((*it)->done(*this))
                {
                    it = receivers_.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // add new receive requests
            std::pair<bool, header> next(acceptor_.next_header());
            if(next.first)
            {
                receivers_.push_back(boost::make_shared<receiver>(next.second, communicator_, *this));
            }

            if(!has_work) has_work = !receivers_.empty();

            if (bootstrapping)
                bootstrapping = hpx::is_starting();

            if(has_work)
            {
                t.restart();
            }
            else
            {
#if defined( WIN32 ) || defined( _WIN32 ) || defined( __WIN32__ ) || defined( __CYGWIN__ )
                Sleep( 1 );
#elif defined( BOOST_HAS_PTHREADS )
                // g++ -Wextra warns on {} or {0}
                struct timespec rqtp = { 0, 0 };

                // POSIX says that timespec has tv_sec and tv_nsec
                // But it doesn't guarantee order or placement

                rqtp.tv_sec = 0;
                rqtp.tv_nsec = 1000;

                nanosleep( &rqtp, 0 );
#endif
            }
        }

        if(stopped_ == true)
        {
            MPI_Comm communicator = communicator_;
            communicator_ = 0;
            MPI_Comm_free(&communicator);
        }
    }
}}}}

#endif
