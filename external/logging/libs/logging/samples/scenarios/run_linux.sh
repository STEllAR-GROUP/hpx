export BOOST_ROOT_DIR="$1"
echo Output is redirected to results.txt
bjam --v2 --boost="$1" --toolset=$2 > results.txt 2>&1

