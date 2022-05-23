reg_str="jacob"
ps -efww | grep ${reg_str} | grep -v "grep" | awk '{print $2}' | xargs kill -9
