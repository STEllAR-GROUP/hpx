
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_VOID_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_VOID_HPP

#include <examples/bright_future/dataflow/dataflow_base_fwd.hpp>
#include <examples/bright_future/dataflow/stubs/dataflow.hpp>

namespace hpx { namespace lcos {
    template <>
    struct dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;
        typedef
            components::client_base<
                dataflow_base<void>
              , stubs::dataflow
            >
            base_type;

        typedef stubs::dataflow stub_type;

        dataflow_base()
            : gid_(naming::invalid_id)
        {}

        virtual ~dataflow_base()
        {
        }

        dataflow_base(promise<naming::id_type, naming::gid_type> const & promise)
            : gid_promise(promise)
            , gid_(naming::invalid_id)
        {
        }

        void get()
        {
            promise<void> p;
            connect(p.get_gid());
            p.get();
        }

        void invalidate()
        {
            gid_ = naming::invalid_id;
            gid_promise.reset();
        }

        void connect(naming::id_type const & target) const
        {
            stub_type::connect(this->get_gid(), target);
        }

        promise<void> connect_async(naming::id_type const & target) const
        {
            return stub_type::connect_async(this->get_gid(), target);
        }

        naming::id_type & get_gid()
        {
            if(!gid_)
            {
                gid_ = gid_promise.get();
            }
            return gid_;
        }

        naming::id_type const & get_gid() const
        {
            if(!gid_)
            {
                gid_ = gid_promise.get();
            }
            return gid_;
        }

    protected:
        promise<naming::id_type, naming::gid_type> gid_promise;
        mutable naming::id_type gid_;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            BOOST_ASSERT(!gid_);
            ar & gid_;
            BOOST_ASSERT(gid_);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            BOOST_ASSERT(this->get_gid());
            ar & this->get_gid();
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER();
    };
}}

#endif
