#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ------------------------------------------------------
# pycicle
# Python Continuous Integration Command Line Engine
# Simple tool to poll PRs on github and spawn builds
# ------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals
import github, os, subprocess, time, re, string, random, socket, datetime

#----------------------------------------------------------------------------
# user's home dir and pycicle root
#----------------------------------------------------------------------------
home = str(os.path.expanduser('~'))
pycicle_dir = os.environ.get('PYCICLE_ROOT', home + '/pycicle')
print('Pycicle using root path', pycicle_dir)

user_token = 'generate a token and paste it here, or set env var'
user_token = os.environ.get('PYCICLE_GITHUB_TOKEN', user_token)

#----------------------------------------------------------------------------
# Debuging - set PYCICLE_DEBUG env var to disable triggering builds
#----------------------------------------------------------------------------
debug_mode = os.environ.get('PYCICLE_DEBUG', '') != ''

#----------------------------------------------------------------------------
# Create a Github instance:
#----------------------------------------------------------------------------
reponame  = 'HPX'
orgname   = 'STEllAR-GROUP'
poll_time = 60

git  = github.Github(orgname, user_token)
org  = git.get_organization(orgname)
repo = org.get_repo(reponame)

#----------------------------------------------------------------------------
# TODO : add support for multiple machines and configs
#----------------------------------------------------------------------------
current_path = os.path.dirname(os.path.realpath(__file__))
nickname = 'greina'
compiler = 'gcc'
boost = '1.65.1'

#----------------------------------------------------------------------------
# read one value from the CMake config for use elsewhere
#----------------------------------------------------------------------------
def get_setting_for_machine(machine, setting) :
    f = open(current_path + '/config/' + machine + '.cmake')
    for line in f:
        m = re.findall(setting + ' \"(.+?)\"', line)
        if m:
            return m[0]
    return ''

#----------------------------------------------------------------------------
# launch a script that will do one build
#----------------------------------------------------------------------------
def launch_pr_build(pr_number, pr_branchname) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
    remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')
    remote_http = get_setting_for_machine(nickname, 'PYCICLE_HTTP')

    #  cmd = ['ssh ' + remote_ssh, 'echo -S ',
    cmd = ['ssh', remote_ssh, 'ctest -S ',
           remote_path          +'/repo/tools/pycicle/dashboard_slurm.cmake',
           '-DPYCICLE_ROOT='    + remote_path,
           '-DPYCICLE_HOST='    + nickname,
           '-DPYCICLE_PR='      + str(pr_number),
           '-DPYCICLE_BRANCH='  + pr_branchname,
           '-DPYCICLE_RANDOM='  + random_string(10),
           '-DPYCICLE_COMPILER='+ compiler,
           '-DPYCICLE_BOOST='   + boost,
           '-DPYCICLE_MASTER='  + 'master',
           # Thes are to quiet warnings from ctest about unset vars
           '-DCTEST_SOURCE_DIRECTORY=.',
           '-DCTEST_BINARY_DIRECTORY=.',
           '-DCTEST_COMMAND=":"']
    if debug_mode:
        print('Debug ', cmd)
    else:
        print('Executing ', cmd)
        p = subprocess.Popen(cmd)
        # print("pid = ", p.pid)

    return None

#----------------------------------------------------------------------------
# random string of N chars
#----------------------------------------------------------------------------
def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

#----------------------------------------------------------------------------
# main polling routine
#----------------------------------------------------------------------------
if debug_mode:
    print('PYCICLE is in debug mode, no build trigger commands will be sent')

t1 = datetime.datetime.now()
master_branch = repo.get_branch(repo.default_branch)
while True:
    #
    try:
        t2 = datetime.datetime.now()
        time_diff = t2-t1
        t1 = t2
        print('Checking github:', 'Time since last check (s)', time_diff.seconds)
        #
        master_sha = master_branch.commit
        #
        for pr in repo.get_pulls('open'):
            if not pr.mergeable:
                continue
            #
            pr_str        = str(pr.number)
            pr_branchname = pr.head.label.rsplit(':',1)[1]
            directory     = pycicle_dir + '/src/' + pr_str
            last_sha      = directory + '/last_pr_sha.txt'
            update        = False
            #
            if not os.path.exists(directory):
                os.makedirs(directory)
                update = True
            else:
                try:
                    f = open(last_sha,'r')
                    lines = f.readlines()
                    if lines[0].strip() != pr.head.sha:
                        print('PR', pr.number, pr_branchname, 'changed : trigger update')
                        update = True
                    if (lines[1].strip() != master_sha.sha):
                        print('master changed : trigger update')
                        update = True
                    f.close()
                except:
                    update = True
            #
            if update:
                f = open(last_sha,'w')
                f.write(pr.head.sha + '\n')
                f.write(master_sha.sha + '\n')
                f.close()
                launch_pr_build(pr.number, pr_branchname)
            else:
    #            print(pr.number, 'is up to date', pr.merge_commit_sha)
                pass
    except (github.GithubException, socket.timeout) as ex:
        # github might be down, or there may be a network issue,
        # just go to the sleep statement and try again in a minute
        print('Github/Socket exception :',ex)
        pass

    # Sleep for a while before polling github again
    time.sleep(poll_time)
