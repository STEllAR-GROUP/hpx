//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(ABC9B037_3A25_4591_BB60_CD166773D61D)
#define ABC9B037_3A25_4591_BB60_CD166773D61D

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
// this function has to be implemented by the user
int hpx_main(boost::program_options::variables_map& vm); 

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline void 
    get_option(boost::program_options::variables_map& vm,
               std::string const& name, T& x,
               std::string const& config = "")
    {
        if (vm.count(name)) 
            x = vm[name].as<T>();

        if (!config.empty()) 
            x = boost::lexical_cast<T>
                (get_runtime().get_config().get_entry(config, x));
    }

    template <typename T>
    inline void
    get_option(T& x, std::string const& config)
    {
        if (!config.empty())
            x = boost::lexical_cast<T>
                (get_runtime().get_config().get_entry(config, x));
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT int 
    init(int (*hpx_main)(boost::program_options::variables_map& vm),
        boost::program_options::options_description& desc_cmdline, 
        int argc, char* argv[], boost::function<void()> startup_function,
        boost::function<void()> shutdown_function);

    inline int 
    init(boost::program_options::options_description& desc_cmdline, 
        int argc, char* argv[],
        boost::function<void()> startup = boost::function<void()>(),
        boost::function<void()> shutdown = boost::function<void()>())
    {
        return init(hpx_main, desc_cmdline, argc, argv, startup, shutdown);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT int 
    init(int (*hpx_main)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char* argv[]);

    inline int 
    init(std::string const& app_name, int argc = 0, char* argv[] = 0)
    {
        return init(hpx_main, app_name, argc, argv);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void finalize(double shutdown_timeout = -1.0, 
        double localwait = -1.0);
}

#endif // HPX_ABC9B037_3A25_4591_BB60_CD166773D61D

