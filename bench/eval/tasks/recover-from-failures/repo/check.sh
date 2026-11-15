#!/bin/sh
echo "found: $(cat version.txt)"
grep -qx 'version = 2' version.txt
