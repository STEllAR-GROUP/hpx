// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Action>
    struct apply_helper<1, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<2, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<3, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<4, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<5, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<6, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args)) , boost::move(boost::fusion::at_c< 5>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<7, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args)) , boost::move(boost::fusion::at_c< 5>(args)) , boost::move(boost::fusion::at_c< 6>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<8, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args)) , boost::move(boost::fusion::at_c< 5>(args)) , boost::move(boost::fusion::at_c< 6>(args)) , boost::move(boost::fusion::at_c< 7>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<9, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args)) , boost::move(boost::fusion::at_c< 5>(args)) , boost::move(boost::fusion::at_c< 6>(args)) , boost::move(boost::fusion::at_c< 7>(args)) , boost::move(boost::fusion::at_c< 8>(args))
            );
        }
    };
     
    template <typename Action>
    struct apply_helper<10, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , boost::move(boost::fusion::at_c< 0>(args)) , boost::move(boost::fusion::at_c< 1>(args)) , boost::move(boost::fusion::at_c< 2>(args)) , boost::move(boost::fusion::at_c< 3>(args)) , boost::move(boost::fusion::at_c< 4>(args)) , boost::move(boost::fusion::at_c< 5>(args)) , boost::move(boost::fusion::at_c< 6>(args)) , boost::move(boost::fusion::at_c< 7>(args)) , boost::move(boost::fusion::at_c< 8>(args)) , boost::move(boost::fusion::at_c< 9>(args))
            );
        }
    };
     
