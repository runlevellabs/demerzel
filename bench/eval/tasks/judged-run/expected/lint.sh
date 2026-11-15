#!/bin/sh
if grep -q "^owner = " mode.txt; then
  echo "lint: ok"
else
  echo "lint: mode.txt has no owner"
  exit 1
fi
