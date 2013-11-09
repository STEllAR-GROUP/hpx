//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_QT_RUNTIME_HPP
#define HPX_QT_RUNTIME_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/util/function.hpp>

#include <QThread>

namespace hpx {
    namespace threads
    {
        struct threadmanager_base;
    }

    namespace qt {

    class runtime : public QObject
    {
        Q_OBJECT

        public:
            struct impl;
            friend struct impl;
            
            runtime(int argc, char ** argv, QObject * parent = 0);

            ~runtime();

            static void apply(HPX_STD_FUNCTION<void()> const& f);

        signals:
            void hpx_started();

        private:
            void qt_startup();
            static void qt_shutdown(impl * impl_);
            impl * impl_;
            static hpx::threads::threadmanager_base * tm_;
    };
}}

#endif
