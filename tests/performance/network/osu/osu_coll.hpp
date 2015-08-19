//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <utility>

#define SKIP 200
#define SKIP_LARGE 10
#define LARGE_MESSAGE_SIZE 8192
#define ITERATIONS_LARGE 100

struct params
{
    std::size_t max_msg_size;
    std::size_t iterations;
    std::size_t fan_out;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & max_msg_size;
        ar & iterations;
        ar & fan_out;
    }
};

boost::program_options::options_description params_desc()
{
    boost::program_options::options_description
        desc("Usage: " HPX_APPLICATION_STRING " [options]");

    desc.add_options()
        ("max-msg-size",
         boost::program_options::value<std::size_t>()->default_value(1048576),
         "Set maximum message size in bytes.")
        ("iter",
         boost::program_options::value<std::size_t>()->default_value(1000),
         "Set number of iterations per message size.")
        ("fan-out",
         boost::program_options::value<std::size_t>()->default_value(2),
         "Set number of iterations per message size.")
        ;

    return desc;
}

params process_args(boost::program_options::variables_map & vm)
{
    params p
        = {
            vm["max-msg-size"].as<std::size_t>()
          , vm["iter"].as<std::size_t>()
          , vm["fan-out"].as<std::size_t>()
        };

    return p;
}

void print_header(std::string const & benchmark)
{
    hpx::cout << "# " << benchmark << hpx::endl
              << "# Size    Latency (microsec)" << hpx::endl
              << hpx::flush;
}

void print_data(double elapsed, std::size_t size, std::size_t iterations)
{
    hpx::cout << std::left << std::setw(10) << size
              << elapsed
              << hpx::endl << hpx::flush;
}

inline std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
distribute_component(std::vector<hpx::id_type> localities,
    hpx::components::component_type type);

HPX_PLAIN_ACTION(distribute_component);

inline std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
distribute_component(std::vector<hpx::id_type> localities,
    hpx::components::component_type type)
{
    typedef hpx::util::remote_locality_result value_type;
    typedef std::pair<std::size_t, std::vector<value_type> > result_type;

    result_type res;
    if(localities.size() == 0) return res;

    hpx::id_type this_loc = localities[0];

    typedef
        hpx::components::server::runtime_support::bulk_create_components_action
        action_type;

    std::size_t num_components = hpx::get_os_thread_count();

    typedef hpx::future<std::vector<hpx::naming::gid_type> > future_type;

    future_type f;
    {
        hpx::lcos::packaged_action<action_type, std::vector<hpx::naming::gid_type> > p;
        p.apply(hpx::launch::async, this_loc, type, num_components);
        f = p.get_future();
    }

    std::vector<hpx::future<result_type> > components_futures;
    components_futures.reserve(2);

    if(localities.size() > 1)
    {
        std::size_t half = (localities.size() / 2) + 1;
        std::vector<hpx::id_type> locs_first(localities.begin()
            + 1, localities.begin() + half);
        std::vector<hpx::id_type> locs_second(localities.begin() + half,
            localities.end());


        if(locs_first.size() > 0)
        {
            hpx::lcos::packaged_action<distribute_component_action, result_type > p;
            hpx::id_type id = locs_first[0];
            p.apply(hpx::launch::async, id, std::move(locs_first), type);
            components_futures.push_back(
                p.get_future()
            );
        }

        if(locs_second.size() > 0)
        {
            hpx::lcos::packaged_action<distribute_component_action, result_type > p;
            hpx::id_type id = locs_second[0];
            p.apply(hpx::launch::async, id, std::move(locs_second), type);
            components_futures.push_back(
                p.get_future()
            );
        }
    }

    res.first = num_components;
    res.second.push_back(
        value_type(this_loc.get_gid(), type)
    );
    res.second.back().gids_ = f.get();

    while(!components_futures.empty())
    {
        hpx::wait_any(components_futures);

        std::size_t ct = 0, pos = 0;

        for (hpx::future<result_type>& f : components_futures)
        {
            if(f.is_ready())
            {
                pos = ct;
                break;
            }
            ++ct;
        }

        result_type r = components_futures.at(pos).get();
        res.second.insert(res.second.end(), r.second.begin(), r.second.end());
        res.first += r.first;
        components_futures.erase(components_futures.begin() + pos);
    }

    return res;
}

template <typename Component>
inline std::vector<hpx::id_type> create_components(params const & p)
{
    hpx::components::component_type type =
        hpx::components::get_component_type<Component>();

    std::vector<hpx::id_type> localities = hpx::find_all_localities(type);

    hpx::id_type id = localities[0];
    hpx::future<std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> > >
        async_result = hpx::async<distribute_component_action>(
            id, std::move(localities), type);

    std::vector<hpx::id_type> components;

    std::pair<std::size_t, std::vector<hpx::util::remote_locality_result> >
        result(async_result.get());

    std::size_t num_components = result.first;
    components.reserve(num_components);

    std::vector<hpx::util::locality_result> res;
    res.reserve(result.second.size());
    for (hpx::util::remote_locality_result const& rl : result.second)
    {
        res.push_back(rl);
    }

    for (hpx::id_type const& id : hpx::util::locality_results(res))
    {
        //hpx::apply<typename Component::run_action>(id, p);
        components.push_back(id);
    }

    return components;
}
