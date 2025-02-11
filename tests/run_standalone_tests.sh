#!/bin/bash

set -e
RED='\033[0;31m'
NC='\033[0m'

# Defaults
sdaas=4
files=""
filter=""
marker="standalone"

# Parse input args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sdaas)
            sdaas="$2"
            shift 2
            ;;
        -f|--files)
            shift
            while [[ "$1" != -* && ! -z "$1" ]]; do
              files+=" $1"
              shift
            done
            ;;
        -k|--filter)
            filter="$2"
            shift 2
            ;;
        -m|--marker)
            marker="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$files" ]]; then
  test_files="tests/test_pytorch"
else
  test_files=$files
fi
echo "Test files: $test_files"

# Get all the tests marked with standalone marker
TEST_FILE="standalone_tests.txt"
test_command="python -um pytest ${test_files} -q --collect-only --pythonwarnings ignore -m \"${marker}\""

if [[ -n "$filter" ]]; then
  test_command+=" -k $filter"
fi
echo "$test_command"
eval "$test_command" > $TEST_FILE
cat $TEST_FILE
sed -i '$d' $TEST_FILE

# Declare an array to store test results
declare -a results

# Get test list and run each test individually
tests=$(grep -oP '^tests/test_\S+' "$TEST_FILE")
for test in $tests; do
  echo "Executing test: $test"
  result=$(python -um pytest -sv "$test" --sdaas $sdaas --pythonwarnings ignore --junitxml="$test"-results.xml)
  retval=$?
  last_line=$(tail -n 1 <<< "$result")

  pattern='([0-9]+) (.*) in ([0-9.]+s)'
  status=""
  if [[ $last_line =~ $pattern ]]; then
      status="${BASH_REMATCH[2]}"
  elif [ "$retval" != 0 ]; then
    echo -e "${RED}$(cat $test-results.xml)${NC}"
    exit 1
  fi

  results+=("${test}:${status}")
done

echo "===== STANDALONE TEST STATUS BEGIN ====="
for result in "${results[@]}"; do
  echo $result
done
echo "===== STANDALONE TEST STATUS END ====="

find tests/ | grep xml | xargs -i mv {} .
rm $TEST_FILE