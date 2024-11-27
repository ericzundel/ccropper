#! env bash
#
# Takes a single directory as an input

DIR="$*"
if [ -d "$DIR" ] ; then
	if [ '$(ls -A "$DIR")' ] ; then
		./ccropper-mult.py -n 5 -border 50 -param1 25 -param2 15 -minradius 500 -maxradius 1500 -noinvert "$DIR"/*.JPG
	else
		echo "Directory empty $DIR"
	fi
else
	echo "Directory not found $DIR"
fi

