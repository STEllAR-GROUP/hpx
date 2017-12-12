#  Copyright (c) 2017 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# pycicle
# Python Continuous Integration Command Line Engine
# Simple tool to poll PRs/etc on github and spawn builds
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import github, os, subprocess, time, re, string, random, socket, datetime, argparse

#--------------------------------------------------------------------------
# Command line args
#--------------------------------------------------------------------------
parser = argparse.ArgumentParser()

#----------------------------------------------
# enable or disable slurm for Job launching
#----------------------------------------------
parser.add_argument('-s', '--slurm', dest='slurm', action='store_true',
    help="Use slurm for job launching (default)")
parser.add_argument('--no-slurm', dest='slurm', action='store_false',
    help="Disable slurm job launching")
parser.set_defaults(slurm=True)

#----------------------------------------------
# enable/debug mode
#----------------------------------------------
parser.add_argument('-d', '--debug', dest='debug', action='store_true',
    default=False, help="Enable debug mode")

#----------------------------------------------
# set default path for pycicle work dir
#----------------------------------------------
home = str(os.path.expanduser('~'))
pycicle_dir = os.environ.get('PYCICLE_ROOT', home + '/pycicle')
parser.add_argument('-r', '--pycicle-dir', dest='pycicle_dir',
    default=pycicle_dir, help='pycicle root path/directory')

#--------------------------------------------------------------------------
# github token used to authenticate access
#--------------------------------------------------------------------------
user_token = 'generate a token and paste it here, or set env var'
user_token = os.environ.get('PYCICLE_GITHUB_TOKEN', user_token)
parser.add_argument('-t', '--github-token', dest='user_token',
    default=user_token, help='github token used to authenticate access')

#--------------------------------------------------------------------------
# Machines : get a list of machines to use for testing (env{PYCICLE_MACHINES})
# use a space separated list of machine nicknames such as
# -m greina daint jb-laptop
# where the names corresond to the name.cmake files in the config dir
#
# TODO : add support for multiple machines and configs
#--------------------------------------------------------------------------
machines = {os.environ.get('PYCICLE_MACHINES', 'greina')}
parser.add_argument('-m', '--machines', dest='machines', nargs='+',
    default=machines, help='list of machines to use for testing')

#--------------------------------------------------------------------------
# PR - when testing, limit checks to a single PR
#--------------------------------------------------------------------------
parser.add_argument('-p', '--pull-request', dest='pull_request', type=int,
    default=0, help='A single PR number for limited testing')

#--------------------------------------------------------------------------
# only enable scraping to test github status setting
#--------------------------------------------------------------------------
parser.add_argument('-c', '--scrape-only', dest='scrape_only', action='store_true',
default=False, help="Only scrape results and set github status (no building)")

#----------------------------------------------
# print summary of parse args
#----------------------------------------------
args = parser.parse_args()
print('pycicle: slurm       :', 'enabled' if args.slurm else 'disabled')
print('pycicle: debug       :',
    'enabled (no build trigger commands will be sent)' if args.debug else 'disabled')
print('pycicle: scrape-only :', 'enabled' if args.slurm else 'disabled')
print('pycicle: path        :', args.pycicle_dir)
print('pycicle: token       :', args.user_token)
print('pycicle: machines    :', args.machines)
print('pycicle: PR          :', args.pull_request)
#
machine = args.machines[0]
print('\ncurrent implementation supports only 1 machine :', machine, '\n')

#--------------------------------------------------------------------------
# Create a Github instance:
#--------------------------------------------------------------------------
reponame    = 'HPX'
orgname     = 'STEllAR-GROUP'
poll_time   = 60
scrape_time = 10*60

try:
    git  = github.Github(orgname, user_token)
    org  = git.get_organization(orgname)
    repo = org.get_repo(reponame)
except:
    print('Failed to connect to github. Network down?')

#--------------------------------------------------------------------------
# Scrape-list : machine/build that we must check for status files
# This will need to support lots of build/machine combinations eventually
#--------------------------------------------------------------------------
scrape_list = {}

#--------------------------------------------------------------------------
# read one value from the CMake config for use elsewhere
#--------------------------------------------------------------------------
def get_setting_for_machine(machine, setting) :
    current_path = os.path.dirname(os.path.realpath(__file__))
    f = open(current_path + '/config/' + machine + '.cmake')
    for line in f:
        m = re.findall(setting + ' \"(.+?)\"', line)
        if m:
            return m[0]
    return ''

#--------------------------------------------------------------------------
# launch a command that will start one build
#--------------------------------------------------------------------------
def launch_build(nickname, compiler, branch_id, branch_name) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
    remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')
    remote_http = get_setting_for_machine(nickname, 'PYCICLE_HTTP')

    # we are not yet using these as 'options'
    boost = 'x.xx.x'
    #
    cmd1 = 'ctest' if not args.debug else 'echo'
    script = 'dashboard_slurm.cmake' if args.slurm else 'dashboard_script.cmake'
    cmd = ['ssh', remote_ssh, cmd1, '-S',
           remote_path          + '/repo/tools/pycicle/' + script,
           '-DPYCICLE_ROOT='    + remote_path,
           '-DPYCICLE_HOST='    + nickname,
           '-DPYCICLE_PR='      + branch_id if branch_id != 'master' else '',
           '-DPYCICLE_BRANCH='  + branch_name,
           '-DPYCICLE_RANDOM='  + random_string(10),
           '-DPYCICLE_COMPILER='+ compiler,
           '-DPYCICLE_BOOST='   + boost,
           '-DPYCICLE_MASTER='  + 'master',
           # These are to quiet warnings from ctest about unset vars
           '-DCTEST_SOURCE_DIRECTORY=.',
           '-DCTEST_BINARY_DIRECTORY=.',
           '-DCTEST_COMMAND=":"']

    if args.debug:
        print('\n' + '-' * 20, 'Debug\n', subprocess.list2cmdline(cmd))
        print('-' * 20 + '\n')
    else:
        print('\n' + '-' * 20, 'Executing\n', subprocess.list2cmdline(cmd))
        p = subprocess.Popen(cmd)
        print('-' * 20 + '\n')

    return None

#--------------------------------------------------------------------------
# launch one build from a list of options
#--------------------------------------------------------------------------
def choose_and_launch(machine, branch_id, branch_name) :
    if machine=='daint':
        if bool(random.getrandbits(1)):
            launch_build(machine, 'gcc', branch_id, branch_name)
        else:
            launch_build(machine, 'clang', branch_id, branch_name)
    else:
        launch_build(machine, 'gcc', branch_id, branch_name)

#--------------------------------------------------------------------------
# find all the PR build jobs submitted and from them the build dirs
# that we can use to scrape results from
#--------------------------------------------------------------------------
def find_scrape_files(nickname) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')
    remote_path = get_setting_for_machine(nickname, 'PYCICLE_ROOT')

    JobFiles   = []
    PR_numbers = {}
    #
    cmd = ['ssh', remote_ssh, 'find ',
           remote_path + '/build/', '-maxdepth 2', '-name pycicle-TAG.txt']
    #
    result = subprocess.check_output(cmd).splitlines()
    for s in result: JobFiles.append(s.decode('utf-8'))

    # for each build dir, return the PR number and results file
    for f in JobFiles:
        m = re.search(r'/build/(.+?)-.*/pycicle-TAG.txt', f)
        if m:
            PR_numbers[m.group(1)] = f

    return PR_numbers

#--------------------------------------------------------------------------
# Utility function to remove a file from a remote filesystem
#--------------------------------------------------------------------------
def erase_file(remote_ssh, scrape_file):
    # erase the pycicle scrape file if we have set status corectly
    try:
        cmd = ['ssh', remote_ssh, 'rm', '-f', scrape_file]
        result = subprocess.check_output(cmd).split()
        print ('File removed', scrape_file)
    except Exception as ex:
        print ('File deletion failed', ex)

#--------------------------------------------------------------------------
# collect test results so that we can update github PR status
#--------------------------------------------------------------------------
def scrape_testing_results(nickname, scrape_file, branch_id, branch_name, head_commit) :
    remote_ssh  = get_setting_for_machine(nickname, 'PYCICLE_MACHINE')

    cmd = ['ssh', remote_ssh, 'cat', scrape_file]

    Config_Errors = 0
    Build_Errors  = 0
    Test_Errors   = 0
    Errors        = []

    context = re.search(r'/build/.*-(.+)/pycicle-TAG.txt', scrape_file)

    if context:
        origin = nickname + '-' + context.group(1)
    else:
        origin = 'unknown'

    try:
        result = subprocess.check_output(cmd).split()
        for s in result: Errors.append(s.decode('utf-8'))
        print('Errors are', Errors)

        Config_Errors = int(Errors[0])
        Build_Errors  = int(Errors[1])
        Test_Errors   = int(Errors[2])
        StatusValid   = True if len(Errors)>3 else False
        if StatusValid:
            DateStamp = Errors[3]
            DateURL   = DateStamp[0:4]+'-'+DateStamp[4:6]+'-'+DateStamp[6:8]
            print('Extracted date as', DateURL)

            URL = ('http://cdash.cscs.ch/index.php?project=HPX' +
                   '&date=' + DateURL +
                   '&filtercount=1' +
                   '&field1=buildname/string&compare1=63&value1=' +
                   branch_id + '-' + branch_name)

            if args.debug:
                print ('Debug github PR status', URL)
            else:
                head_commit.create_status(
                    'success' if Config_Errors==0 else 'failure',
                    target_url=URL,
                    description='errors ' + Errors[0],
                    context='pycicle ' + origin + ' Config')
                head_commit.create_status(
                    'success' if Build_Errors==0 else 'failure',
                    target_url=URL,
                    description='errors ' + Errors[1],
                    context='pycicle ' + origin + ' Build')
                head_commit.create_status(
                    'success' if Test_Errors==0 else 'failure',
                    target_url=URL,
                    description='errors ' + Errors[2],
                    context='pycicle ' + origin + ' Test')
                print ('Done setting github PR status for', origin)

        erase_file(remote_ssh, scrape_file)

    except Exception as ex:
        print('Scrape failed for PR', branch_id, ex)

#--------------------------------------------------------------------------
# random string of N chars
#--------------------------------------------------------------------------
def random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
        for _ in range(N))

#--------------------------------------------------------------------------
# Check if a PR Needs and Update
#--------------------------------------------------------------------------
def needs_update(branch_id, branch_name, branch_sha, master_sha):
    directory     = pycicle_dir + '/src/' + branch_id
    status_file   = directory + '/last_pr_sha.txt'
    update        = False
    #
    if not os.path.exists(directory):
        os.makedirs(directory)
        update = True
    else:
        try:
            f = open(status_file,'r')
            lines = f.readlines()
            if lines[0].strip() != branch_sha:
                print(branch_id, branch_name, 'changed : trigger update')
                update = True
            elif (lines[1].strip() != master_sha):
                print('master changed : trigger update')
                update = True
            f.close()
        except:
            print(branch_id, branch_name, 'status error : trigger update')
            update = True
    #
    if update:
        f = open(status_file,'w')
        f.write(branch_sha + '\n')
        f.write(master_sha + '\n')
        f.close()
    #
    return update

#--------------------------------------------------------------------------
# main polling routine
#--------------------------------------------------------------------------
#
github_t1       = datetime.datetime.now()
scrape_t1       = github_t1 + datetime.timedelta(hours=-1)
#
random.seed(7)
#
while True:
    #
    try:
        github_t2     = datetime.datetime.now()
        github_tdiff  = github_t2 - github_t1
        github_t1     = github_t2
        print('Checking github:', 'Time since last check', github_tdiff.seconds, '(s)')
        #
        master_branch = repo.get_branch(repo.default_branch)
        master_sha    = master_branch.commit.sha
        #
        pull_requests = repo.get_pulls('open')
        pr_list       = {}
        for pr in pull_requests:
            #
            branch_id   = str(pr.number)
            branch_name = pr.head.label.rsplit(':',1)[1]
            branch_sha  = pr.head.sha
            # need details, including last commit on PR for setting status
            pr_list[branch_id] = [machine, branch_name, pr.get_commits().reversed[0]]
            #
            if args.pull_request!=0 and pr.number!=args.pull_request:
                continue
            if not pr.mergeable:
                continue
            #
            if not args.scrape_only:
                update = needs_update(branch_id, branch_name, branch_sha, master_sha)
                if update:
                    choose_and_launch(machine, branch_id, branch_name)

        # also build the master branch if it has changed
        if not args.scrape_only and args.pull_request==0:
            if needs_update('master', 'master', master_sha, master_sha):
                choose_and_launch(machine, 'master', 'master')
                pr_list['master'] = [machine, 'master', master_branch.commit, ""]

        scrape_t2    = datetime.datetime.now()
        scrape_tdiff = scrape_t2 - scrape_t1
        if (scrape_tdiff.seconds > scrape_time):
            scrape_t1 = scrape_t2
            print('Scraping results:', 'Time since last check', scrape_tdiff.seconds, '(s)')
            builds_done = find_scrape_files(machine)
            print(builds_done)
            for branch_id in builds_done:
                if branch_id in pr_list:
                    # nickname, scrape_file, branch_id, branch_name, head_commit
                    scrape_testing_results(
                        pr_list[branch_id][0], builds_done.get(branch_id),
                        branch_id, pr_list[branch_id][1], pr_list[branch_id][2])
                else:
                    # just delete the file, it is probably an old one
                    erase_file(
                        get_setting_for_machine(machine, 'PYCICLE_MACHINE'),
                        builds_done.get(branch_id))


    except (github.GithubException, socket.timeout) as ex:
        # github might be down, or there may be a network issue,
        # just go to the sleep statement and try again in a minute
        print('Github/Socket exception :', ex)

    # Sleep for a while before polling github again
    time.sleep(poll_time)
