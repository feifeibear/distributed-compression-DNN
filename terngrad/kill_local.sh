#!/bin/bash
set -x
USER=fangjr
threadid=$( ps aux | grep python | grep ${USER} | awk '{print $2}')
if [[ "$threadid" =~ ^-?[0-9]+.*$ ]] ; 
then
  kill $threadid
else
  echo "Stopped already."
fi
