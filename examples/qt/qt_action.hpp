
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <QObject>

#include <boost/function.hpp>

namespace helper
{
class connect_functor_helper : public QObject {
    Q_OBJECT
public:
    connect_functor_helper(QObject *parent, const boost::function<void(std::size_t)> &f) : QObject(parent), function_(f) {
    }
 
public Q_SLOTS:
    void signaled(std::size_t i) {
        function_(i);
    }
 
private:
    boost::function<void(std::size_t)> function_;
};

template <class T>
bool connect(QObject *sender, const char *signal, const T &reciever, Qt::ConnectionType type = Qt::AutoConnection) {
    return QObject::connect(sender, signal, new connect_functor_helper(sender, reciever), SLOT(signaled()), type);
}

}
