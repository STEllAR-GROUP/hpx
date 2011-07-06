./bh -i 100000_file
mv gmon.out gmon.sum
./bh -i 100000_file
gprof -s bh gmon.out gmon.sum
gprof bh gmon.sum > prof_100000
rm -rf gmon.out
rm -rf gmon.sum
