//  Copyright (c) 2012 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_amcKqeEjd01sNoYXrWLZkVt5Il2vwxFzhT9f78uUnAgBPCpJbDRQ)
#define HPX_amcKqeEjd01sNoYXrWLZkVt5Il2vwxFzhT9f78uUnAgBPCpJbDRQ

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <cstdio>

namespace distributed

{
    //Information for component configuration
    struct HPX_COMPONENT_EXPORT config_comp
    { 
        config_comp();//: num_instances_(0) {}
        config_comp(std::string const& symbolic_name, std::size_t num_instances
            , std::size_t my_cardinality);
        ~config_comp();

        std::string symbolic_name_;  // symbolic name this instance is registered
        std::size_t num_instances_;
        std::size_t my_cardinality_;
    };
    
}

namespace distributed { namespace server
{

    class HPX_COMPONENT_EXPORT datastructure
        : public hpx::components::simple_component_base<datastructure>
    {
    public: 
    
        enum actions
        {
            datastructure_init = 0,
            datastructure_get_data = 1,
            datastructure_write = 2,
            datastructure_config_get = 3,
            datastructure_get_data_at = 4
            //datastructure_config_init = 4;
        };
    
        datastructure();// {}
        ~datastructure();//{}

        typedef std::vector<std::size_t> data_type;

        void data_init(std::string const& symbolic_name
            , std::size_t num_instances
            , std::size_t my_cardinality
            ,std::size_t const init_length
            , std::size_t const init_value);

        void data_write(std::string const& symbolic_name
            , std::size_t num_instances
            , std::size_t my_cardinality
            , data_type client_data);

        config_comp get_config_info() const;

        data_type get_data();

        std::size_t get_data_at(std::size_t pos);

        typedef hpx::actions::result_action1<
        datastructure
        , std::size_t
        , datastructure_get_data_at
        , std::size_t
        , &datastructure::get_data_at
        > get_data_at_action;

        typedef hpx::actions::result_action0<
        datastructure
        , data_type
        , datastructure_get_data
        , &datastructure::get_data
        > get_data_action;

        typedef hpx::actions::result_action0<
        datastructure const
        , distributed::config_comp
        , datastructure_config_get
        , &datastructure::get_config_info
        > get_config_action;

        typedef hpx::actions::action4<
        datastructure
        , datastructure_write
        , std::string const&
        , std::size_t
        , std::size_t
        , data_type 
        , &datastructure::data_write
        > write_action;

        typedef hpx::actions::action5<
        datastructure
        , datastructure_init
        , std::string const&
        , std::size_t
        , std::size_t
        , std::size_t
        , std::size_t
        , &datastructure::data_init       
        > init_action;

    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & config_data_;
            ar & data_;
        }

    private:
        config_comp config_data_;   //configure data
        std::vector<size_t> data_;  //simple vector data
    };
}}
////////////////////////////////////////////////////////////////////////////////
//Non intrusive serialization
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive&, distributed::config_comp&, unsigned int const);
}}

HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::init_action,
    distributed_datastructure_init_action);
HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::write_action,
    distributed_datastructure_write_action);
HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::get_config_action,
    distributed_datastructure_get_config_action);
HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::get_data_action,
    distributed_datastructure_get_data_action);
HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::get_data_at_action,
    distributed_datastructure_get_data_at_action);

//HPX_REGISTER_ACTION_DECLARATION_EX(distributed::server::datastructure::
    //get_config_action, distributed_datastructure_get_config_action);
//HPX_REGISTER_ACTION_DECLARATION_EX(
//    hpx::lcos::base_lco_with_value<distributed::config_comp>::set_result_action,
//    set_result_action_distributed_config_comp);
//
#endif //HPX_amcKqeEjd01sNoYXrWLZkVt5Il2vwxFzhT9f78uUnAgBPCpJbDRQ
