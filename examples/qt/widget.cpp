
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "widget.hpp"
#include "hpx_qt.hpp"

#include <QtGui/QLabel>
#include <QtGui/QHBoxLayout>
#include <QtGui/QVBoxLayout>
#include <QtGui/QSpinBox>
#include <QtGui/QPushButton>
#include <QtGui/QListWidget>

widget::widget(boost::function<void(widget *, std::size_t)> callback, QWidget *parent)
    : QDialog(parent)
    , no_threads(50)
    , callback_(callback)
{
    QHBoxLayout * layout = new QHBoxLayout;

    QSpinBox * thread_number_widget = new QSpinBox;
    thread_number_widget->setValue(50);
    thread_number_widget->setRange(1, 100000);
    QObject::connect(thread_number_widget, SIGNAL(valueChanged(int)), this, SLOT(set_threads(int)));

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

void widget::add_label(std::size_t i, double t)
{
    QMutexLocker lk(&mutex);
    QString txt("Thread ");
    txt.append(QString::number(i))
       .append(" finished in ")
       .append(QString::number(t))
       .append(" seconds");
    list->addItem(txt);
}

void widget::run_finished()
{
    run_button->setEnabled(true);
}

void widget::set_threads(int no)
{
    no_threads = std::size_t(no);
}

void widget::run_clicked(bool)
{
    {
        run_button->setEnabled(false);
        list->clear();
    }
    
    hpx::qt::runtime::apply(boost::bind(callback_, this, no_threads));
}
