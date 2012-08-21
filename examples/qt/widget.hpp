
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <QtGui/QDialog>
#include <QMutex>

#include <boost/function.hpp>

class QWidget;
class QListWidget;
class QPushButton;

class widget
    : public QDialog
{
    Q_OBJECT

    public:
        widget(boost::function<void(widget *, std::size_t)> callback, QWidget *parent = 0);

        void add_label(std::size_t i, double t);

        void run_finished();
    
    public slots:
        void set_threads(int no);

        void run_clicked(bool);

    private:
        std::size_t no_threads;
        QMutex mutex;
        QListWidget *list;
        QPushButton * run_button;
        boost::function<void(widget *, std::size_t)> callback_;
};
