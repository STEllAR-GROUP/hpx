//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/acceptor.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/sender.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/receiver.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines +=
            "buffer_size = ${HPX_PARCEL_IBVERBS_BUFFER_SIZE:65536}",
            "io_pool_size = 1",
            "use_io_pool = 1"
            ;

        return lines;
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , stopped_(false)
      , handling_messages_(false)
      , use_io_pool_(true)
    {
        // we never do zero copy optimization for this parcelport
        allow_zero_copy_optimizations_ = false;
        
        std::string use_io_pool =
            ini.get_entry("hpx.parcel.mpi.use_io_pool", "1");
        if(boost::lexical_cast<int>(use_io_pool) == 0)
        {
            use_io_pool_ = false;
        }
    }

    connection_handler::~connection_handler()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
    }

    bool connection_handler::do_run()
    {
        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it =
                accept_begin(here_, io_service_pool_.get_io_service(0));
             it != end; ++it, ++tried)
        {
            try {
                boost::asio::ip::tcp::endpoint ep = *it;
                
                acceptor_.bind(ep, boost::system::throws);
                ++tried;
                break;
            }
            catch (boost::system::system_error const& e) {
                errors.add(e);   // store all errors
                continue;
            }
        }

        if (errors.get_error_count() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "ibverbs::connection_handler::run", errors.get_message());
            return false;
        }
        background_work();
        return true;
    }

    void connection_handler::do_stop()
    {
        // Mark stopped state
        stopped_ = true;
        // Wait until message handler returns
        std::size_t k = 0;

        // cancel all pending accept operations
        boost::system::error_code ec;
        acceptor_.close(ec);

        while(handling_messages_)
        {
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }
    }

    // Make sure all pending requests are handled
    void connection_handler::background_work()
    {
        if (stopped_)
            return;

        // Atomically set handling_messages_ to true, if another work item hasn't
        // started executing before us.
        bool false_ = false;
        if (!handling_messages_.compare_exchange_strong(false_, true))
            return;

        if(!hpx::is_starting() && !use_io_pool_)
        {
            hpx::applier::register_thread_nullary(
                HPX_STD_BIND(&connection_handler::handle_messages, this),
                "mpi::connection_handler::handle_messages",
                threads::pending, true, threads::thread_priority_critical);
        }
        else
        {
            boost::asio::io_service& io_service = io_service_pool_.get_io_service();
            io_service.post(HPX_STD_BIND(&connection_handler::handle_messages, this));
        }
    }

    std::string connection_handler::get_locality_name() const
    {
        return "ibverbs";
    }

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service();
        boost::shared_ptr<sender> sender_connection(new sender(*this, l, parcels_sent_));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try {
                naming::locality::iterator_type end = connect_end(l);
                for (naming::locality::iterator_type it =
                        connect_begin(l, io_service);
                      it != end; ++it)
                {
                    boost::asio::ip::tcp::endpoint const& ep = *it;

                    client_context& ctx = sender_connection->context();
                    ctx.close(ec);
                    ctx.connect(ep, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend(hpx::threads::pending,
                        "connection_handler(ibverbs)::create_connection");
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                sender_connection->context().close(ec);
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "ibverbs::parcelport::get_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->context().close(ec);
            sender_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "ibverbs::parcelport::get_connection",
                hpx::util::osstream_get_string(strm));
            return sender_connection;
        }

        if (&ec != &throws)
            ec = make_success_code();
        
        return sender_connection;
    }
    
    void connection_handler::add_sender(
        boost::shared_ptr<sender> const& sender_connection)
    {
        hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
        senders_.push_back(sender_connection);
    }

    void add_sender(connection_handler & handler,
        boost::shared_ptr<sender> const& sender_connection)
    {
        handler.add_sender(sender_connection);
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
        std::size_t k = 0;

        hpx::util::high_resolution_timer t;
        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work) || (!has_work && t.elapsed() < 2.0))
        {
            // handle all sends ...
            {
                hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
                for(
                    senders_type::iterator it = senders_.begin();
                    !stopped_ && enable_parcel_handling_ && it != senders_.end();
                    /**/)
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

            // handle all receives ...
            for(
                receivers_type::iterator it = receivers_.begin();
                !stopped_ && enable_parcel_handling_ && it != receivers_.end();
                /**/)
            {
                try
                {
                    if((*it)->done(*this))
                    {
                        (*it)->async_read();
                    }
                }
                catch(boost::system::system_error const& e)
                {
                    if(e.code() == boost::asio::error::eof
                    || e.code() == boost::asio::error::operation_aborted)
                    {
                        it = receivers_.erase(it);
                        continue;
                    }
                    throw;
                }
                ++it;
            }

            // handle all accepts ...
            boost::shared_ptr<receiver> rcv = acceptor_.accept(*this, boost::system::throws);
            if(rcv)
            {
                rcv->async_read();
                receivers_.push_back(rcv);
            }

            if (bootstrapping)
                bootstrapping = hpx::is_starting();

            if(has_work)
            {
                t.restart();
                k = 0;
            }
            else
            {
                if(enable_parcel_handling_)
                {
                    hpx::lcos::local::spinlock::yield(k);
                    ++k;
                }
            }
        }

        if(stopped_ == true)
        {
        }
    }
}}}}

#endif
