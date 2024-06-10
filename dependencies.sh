#!/bin/bash

echo "Installing required libraries using Homebrew..."

if test ! $(which brew); then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

brew update

echo "Installing Eigen..."
brew install eigen

echo "Installing openCV"
brew install opencv

echo "Installing HDF5..."
brew install hdf5

echo "Installing GoogleTest..."
brew install googletest

echo "All required libraries installed successfully."

echo "Creating symbolic links..."

brew link --force --overwrite eigen

brew link --force --overwrite opencv

brew link --force --overwrite hdf5

brew link --force --overwrite googletest

echo "Symbolic links created successfully."

