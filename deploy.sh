#!/bin/sh

# Deploying blog for my small version of SurfaceEvolver for caveolin 3D
# membrane shaping mechanism
# 
# Usage: add custom message after ./deploy 

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Go To Public folder
pwd

# Add changes to git.
git add .

# Commit changes.
msg="uploading updates $(date)"	# default deployment message
if [ -n "$*" ]; then
		msg="$*"
	fi
	echo $msg
	git commit -m "$msg"

	# Push source and build repos.
	git push origin main 
