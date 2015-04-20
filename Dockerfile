FROM stellargroup/build_env:debian_clang

ADD . /hpx

RUN cd /hpx/build && make install && cd /
RUN rm -rf /hpx

RUN ldconfig

CMD bash
WORKDIR /
