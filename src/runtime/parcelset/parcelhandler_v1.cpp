//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/exception.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/runtime/naming/detail/resolver_do_undo.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>

#include <string>
#include <algorithm>

#include <boost/version.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/lambda/lambda.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // A parcel is submitted for transport at the source locality site to 
    // the parcel set of the locality with the put-parcel command
    // This function is synchronous.

    struct wait_for_put_parcel
    {
        typedef boost::mutex mutex_type;
        typedef boost::condition condition_type;

        wait_for_put_parcel(mutex_type& mtx, condition_type& cond,
              boost::system::error_code& saved_error, 
              bool& waiting, bool& finished)
          : mtx_(mtx), cond_(cond), saved_error_(saved_error),
            waiting_(waiting), finished_(finished)
        {}

        void operator()(boost::system::error_code const& e, std::size_t size)
        {
            mutex_type::scoped_lock l(mtx_);
            if (e) 
                saved_error_ = e;

            if (waiting_)
                cond_.notify_one();
            finished_ = true;
        }

        bool wait()
        {
            mutex_type::scoped_lock l(mtx_);

            if (finished_) 
                return true;

            boost::xtime xt;
            boost::xtime_get(&xt, boost::TIME_UTC);
            xt.sec += 5;        // wait for max. 5sec

            waiting_ = true;
            return cond_.timed_wait(l, xt);
        }

        mutex_type& mtx_;
        condition_type& cond_;
        boost::system::error_code& saved_error_;
        bool& waiting_;
        bool& finished_;
    };

    void parcelhandler::sync_put_parcel(parcel& p)
    {
        wait_for_put_parcel::mutex_type mtx;
        wait_for_put_parcel::condition_type cond;
        boost::system::error_code saved_error;
        bool waiting = false, finished = false;

        wait_for_put_parcel wfp(mtx, cond, saved_error, waiting, finished);
        put_parcel(p, wfp);  // schedule parcel send
        if (!wfp.wait())     // wait for the parcel being sent
            HPX_THROW_EXCEPTION(network_error
              , "parcelhandler::sync_put_parcel"
              , "synchronous parcel send timed out");

        if (saved_error) 
            HPX_THROW_EXCEPTION(network_error
              , "parcelhandler::sync_put_parcel"
              , saved_error.message()); 
    }

    void parcelhandler::parcel_sink(parcelport& pp, 
        boost::shared_ptr<std::vector<char> > const& parcel_data,
        threads::thread_priority priority)
    {
        if (NULL == tm_ || !(tm_->status() == running))
        {
            // this is supported for debugging purposes mainly, it results in
            // the direct execution of the parcel decoding
            decode_parcel(parcel_data);
        }

        else 
        {
            // create a new thread which decodes and handles the parcel
            threads::thread_init_data data(
                boost::bind(&parcelhandler::decode_parcel, this, parcel_data),
                "decode_parcel", 0, priority);
            tm_->register_thread(data);
        }
    }

    threads::thread_state parcelhandler::decode_parcel(
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
        // protect from unhandled exceptions bubbling up into thread manager
        try {
            parcel p;
            {
                // create a special io stream on top of in_buffer_
                typedef util::container_device<std::vector<char> > io_device_type;
                boost::iostreams::stream<io_device_type> io(*parcel_data.get());

                // De-serialize the parcel data
#if HPX_USE_PORTABLE_ARCHIVES != 0
                util::portable_binary_iarchive archive(io);
#else
                boost::archive::binary_iarchive archive(io);
#endif
                archive >> p;
            }

            // add parcel to incoming parcel queue
            parcels_->add_parcel(p);
        }
        catch (hpx::exception const& e) {
            LPT_(error) 
                << "Unhandled exception while executing decode_parcel: "
                << e.what();
        }
        return threads::thread_state(threads::terminated);
    }

    bool parcelhandler::get_raw_remote_prefixes(
        std::vector<naming::gid_type>& prefixes, 
        components::component_type type) const
    {
        std::vector<naming::gid_type> allprefixes;
        error_code ec;
        bool result = resolver_.get_prefixes(allprefixes, type, ec);
        if (ec || !result) return false;

        using boost::lambda::_1;
        std::remove_copy_if(allprefixes.begin(), allprefixes.end(), 
            std::back_inserter(prefixes), _1 == prefix_);
        return !prefixes.empty();
    }

    bool parcelhandler::get_raw_prefixes(
        std::vector<naming::gid_type>& prefixes, 
        components::component_type type) const
    {
        error_code ec;
        bool result = resolver_.get_prefixes(prefixes, type, ec);
        if (ec || !result) return false;

        return !prefixes.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    // this handler is called whenever the parcel has been sent
    void release_do_undo(boost::system::error_code const& e, 
            std::size_t size, parcelhandler::write_handler_type f,
            boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo)
    {
        if (e)
            do_undo->undo();
        f(e, size);   // call supplied handler in any case
    }

    ///////////////////////////////////////////////////////////////////////////
    // prepare the given gid, note: this function modifies the passed id
    void prepare_gid(
        boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo, 
        naming::id_type const& id)
    {
        // request new credits from AGAS if needed (i.e. gid is credit managed
        // and the new gid has no credits after splitting)
        boost::uint16_t oldcredits = id.get_credit();
        if (0 != oldcredits) 
        {
            naming::id_type newid(id.split_credits());
            if (0 == newid.get_credit())
            {
                BOOST_ASSERT(1 == id.get_credit());
                do_undo->incref(id, HPX_INITIAL_GLOBALCREDIT, id, oldcredits);
                do_undo->incref(newid, HPX_INITIAL_GLOBALCREDIT, id, oldcredits);
            }
            const_cast<naming::id_type&>(id) = newid;
        }
    }

    // prepare all GIDs stored in an action
    void prepare_action(
        boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo, 
        actions::action_type action)
    {
        action->enumerate_argument_gids(boost::bind(prepare_gid, do_undo, _1));
    }

    // prepare all GIDs stored in a continuation
    void prepare_continuation(
        boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo, 
        actions::continuation_type cont)
    {
        if (cont)
            cont->enumerate_argument_gids(boost::bind(prepare_gid, do_undo, _1));
    }

    // walk through all data in this parcel and register all required AGAS 
    // operations with the given resolver_helper
    void prepare_parcel(
        boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo, 
        parcel& p, error_code& ec)
    {
        prepare_gid(do_undo, p.get_source());                // we need to register the source gid
        prepare_action(do_undo, p.get_action());             // all gids stored in the action
        prepare_continuation(do_undo, p.get_continuation()); // all gids in the continuation
        if (&ec != &throws)
            ec = make_success_code();
    }

    void parcelhandler::put_parcel(parcel& p, write_handler_type f)
    {
        // properly initialize parcel
        init_parcel(p);

        // asynchronously resolve destination address, if needed
        boost::shared_ptr<naming::detail::bulk_resolver_helper> do_undo(
            new naming::detail::bulk_resolver_helper(resolver_));

        if (!p.get_destination_addr())
        { 
            naming::address addr;
            if (!resolver_.resolve_cached(p.get_destination(), addr)) {
                // resolve the remote address
                do_undo->resolve(p.get_destination(), p);
            }
            else {
                p.set_destination_addr(addr);
            }
        }

        // prepare all additional AGAS related operations for this parcel
        error_code ec;
        prepare_parcel(do_undo, p, ec);
        if (ec) {
            // parcel preparation failed
            HPX_THROW_EXCEPTION(no_success, 
                "parcelhandler::put_parcel", ec.get_message());
        }

        // execute all necessary AGAS operations (if any)
        do_undo->execute(ec);
        if (ec) {
            // one or more AGAS operations failed
            HPX_THROW_EXCEPTION(no_success, 
                "parcelhandler::put_parcel", ec.get_message());
        }

        // send the parcel to its destination, return parcel id of the 
        // parcel being sent
        pp_.put_parcel(p, boost::bind(&release_do_undo, _1, _2, f, do_undo));
    }

    ///////////////////////////////////////////////////////////////////////////
    parcelhandler::parcelhandler(naming::resolver_client& resolver, 
            parcelport& pp, threads::threadmanager_base* tm, 
            parcelhandler_queue_base* policy)
      : resolver_(resolver), pp_(pp), tm_(tm), parcels_(policy),
        startup_time_(util::high_resolution_timer::now()), timer_()
    {
        BOOST_ASSERT(parcels_);

        // retrieve the prefix to be used for this site
        resolver_.get_prefix(pp.here(), prefix_); // throws on error

        parcels_->set_parcelhandler(this);

        // register our callback function with the parcelport
        pp_.register_event_handler
            (boost::bind(&parcelhandler::parcel_sink, this, _1, _2, _3));
    }

///////////////////////////////////////////////////////////////////////////////
}}

#endif // HPX_AGAS_VERSION
