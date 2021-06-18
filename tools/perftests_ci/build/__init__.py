# -*- coding: utf-8 -*-

import os

from pyutils import env, log, runtools


def cmake(source_dir, build_dir, install_dir=None):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f'Source directory "{source_dir}" not found')

    source_dir = os.path.abspath(source_dir)

    build_dir = os.path.abspath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    if install_dir is not None:
        install_dir = os.path.abspath(install_dir)
        os.makedirs(install_dir, exist_ok=True)
        env.set_cmake_arg('CMAKE_INSTALL_PREFIX', install_dir)
    command = ['cmake', source_dir] + env.cmake_args()
    runtools.run(command, log_output=log.info, cwd=build_dir)


def make(build_dir, targets=None):
    command = env.build_command()
    if targets is not None:
        command += list(targets)

    runtools.run(command, cwd=build_dir)
