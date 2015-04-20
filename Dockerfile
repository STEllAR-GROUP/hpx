FROM stellargroup/build_env:debian_clang

COPY . /hpx

RUN cd /hpx/build && make install

RUN rm -rf /hpx

RUN ldconfig
