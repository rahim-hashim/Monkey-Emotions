#!/bin/bash
# change directory to the directory with the data
# cd "/Users/rahimhashim/Google Drive/My Drive/Columbia/Salzman/Monkey-Training/tasks/rhAirpuff/8. Probabilistic_Reward_Airpuff_Choice"
cd "/Users/rahimhashim/Google Drive/My Drive/Columbia/Salzman/Monkey-Training/tasks/rhAirpuff/9. Reward_Airpuff_Generalization"
echo "Current directory is:"
echo "  $PWD"
# set target_path to the path of the directory with all the data
# target_path="data_Probabilistic_Reward_Airpuff_Choice/"
target_path="data_Probabilistic_Reward_Airpuff_Generalization"
echo "Target path is:"
echo "  $target_path"

# set monkey variable to Aragorn
monkey="Aragorn"
# set date variable to todays date (i.e. YYMMDD)
# date_str=$(date +%y%m%d)
date_str='231114'
if [ -z "$monkey" ]; then
  echo "Searching for all monkeys for $date_str"
else
  echo "Searching for $monkey for $date_str"
fi
# find file that has the monkey name and todays date using loop and add to list and ends with .h5
file_array=()
for file in *; do
  if [ -z "$monkey" ]; then
    if [[ $file == *"$date_str"* ]] && [[ $file == *".h5"* ]]; then
        file_array+=($file)
    fi
  else
    if [[ $file == *"$monkey"* ]] && [[ $file == *"$date_str"* ]] && [[ $file == *".h5"* ]]; then
        file_array+=($file)
    fi
  fi
done
# if file_array is larger than 10, print warning and exit
if [ ${#file_array[@]} -gt 10 ]; then
  echo "WARNING: More than 10 files found"
  echo "  ${#file_array[@]} files found"
  echo "  Exiting"
  exit
fi
if [ ${#file_array[@]} -eq 0 ]; then
  echo "  No files found"
else
  for file_name in "${file_array[@]}"; do
    echo "  Moving : $file_name"
    # move the data file to the target path
    mv $file_name $target_path
    echo "  Moved  : $file_name"
    done
fi
# print total number of files moved
echo "  Total number of files moved: ${#file_array[@]}"

# copy fractal date folder to all fractals folder
# date_str variable in YYYYMMDD format
fractal_date=$(date -j -f "%y%m%d" "$date_str" +"%Y%m%d")
fractal_folder="_fractals/$fractal_date"
echo "Fractal folder is: $fractal_date"
# copy fractal folder to all fractals folder
cp -r $fractal_folder "../_fractals_all"
echo "  Copied fractal folder to all fractals folder"
echo "Done."