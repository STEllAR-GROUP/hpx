//  Copyright (c) 2012-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <QtGui/QDialog>

#if !defined(Q_MOC_RUN)
#include <hpx/include/threads.hpp>
#include <hpx/include/lcos_local.hpp>
#include <cstddef>
#include <functional>
#endif

class QWidget;
class QListWidget;
class QPushButton;

class widget
    : public QDialog
{
    Q_OBJECT

    public:
        widget(std::function<void(widget *, std::size_t)> callback,
            QWidget *parent = nullptr);

        void threadsafe_add_label(std::size_t i, double t);

        void threadsafe_run_finished();

    public slots:
        void set_threads(int no);

        void run_clicked(bool);

        void add_label(const QString& text);

    private:
        std::size_t no_threads;
        hpx::lcos::local::spinlock mutex;
        QListWidget *list;
        QPushButton * run_button;
        std::function<void(widget *, std::size_t)> callback_;
};
