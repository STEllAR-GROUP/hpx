//  Copyright (c) 2012-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/future.hpp>

#include <cstddef>
#include <functional>
#include <mutex>

#include "widget.hpp"

#include <QtGui/QLabel>
#include <QtGui/QHBoxLayout>
#include <QtGui/QVBoxLayout>
#include <QtGui/QSpinBox>
#include <QtGui/QPushButton>
#include <QtGui/QListWidget>

widget::widget(std::function<void(widget *, std::size_t)> callback, QWidget *parent)
    : QDialog(parent)
    , no_threads(50)
    , callback_(callback)
{
    QHBoxLayout * layout = new QHBoxLayout;

    QSpinBox * thread_number_widget = new QSpinBox;
    thread_number_widget->setValue(50);
    thread_number_widget->setRange(1, 100000);
    QObject::connect(thread_number_widget, SIGNAL(valueChanged(int)),
        this, SLOT(set_threads(int)));

    run_button = new QPushButton("Run");
    QObject::connect(run_button, SIGNAL(clicked(bool)), this, SLOT(run_clicked(bool)));

    layout->addWidget(new QLabel("Number of threads: "));
    layout->addWidget(thread_number_widget);
    layout->addWidget(run_button);

    QVBoxLayout * main_layout = new QVBoxLayout;
    main_layout->addLayout(layout);

    list = new QListWidget;
    main_layout->addWidget(list);

    setLayout(main_layout);
}

void widget::threadsafe_add_label(std::size_t i, double t)
{
    // may be called from any thread, doesn't interact directly with GUI objects
    QString txt("Thread ");
    txt.append(QString::number(i))
       .append(" finished in ")
       .append(QString::number(t))
       .append(" seconds");

    QGenericArgument arg("const QString&", &txt);
    // Qt::QueuedConnection makes sure 'add_label' is called in the GUI thread
    QMetaObject::invokeMethod(this, "add_label", Qt::QueuedConnection, arg);
}

void widget::threadsafe_run_finished()
{
    // may be called from any thread, does not interact directly with GUI objects
    bool value = true;
    QGenericArgument arg("bool",&value);
    // Qt::QueuedConnection makes sure 'setEnabled' is called in the GUI thread
    QMetaObject::invokeMethod(
                run_button, "setEnabled", Qt::QueuedConnection, arg);
}

void widget::set_threads(int no)
{
    no_threads = std::size_t(no);
}

void widget::run_clicked(bool)
{
    run_button->setEnabled(false);
    list->clear();
    hpx::apply(callback_, this, no_threads);
}

void widget::add_label(const QString &text)
{
    list->addItem(text);
}
