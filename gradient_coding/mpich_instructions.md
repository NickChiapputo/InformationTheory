# How to Install MPICH
1. Navigate to https://www.mpich.org/
	1. Documentation and guides are found at https://www.mpich.org/documentation/guides/
	1. For installation guide, click on "MPICH Installers' Guide" from the above link. The actual link changes with the version. Version 3.4 is https://www.mpich.org/static/downloads/3.4/mpich-3.4-installguide.pdf.
	1. The basic instructions are found in the README. For 3.4: https://www.mpich.org/static/downloads/3.4/mpich-3.4-README.txt
1. From https://www.mpich.org/downloads/, download the current stable version. Currently, this is 3.4. File name is mpi-<ver>.tar.gz
	1. Requirements for installation:
		* C compiler
1. Unpack the tar file: `tar xfz mpich-<ver>.tar.gz`
	1. This will unpack the files into a folder titled mpich-<ver>
1. If desired, create the installation directory.
	1. Use the command `mkdir /home/you/mpich-install`
	1. Otherwise, the next steps will create the installation directory in the /usr/local/bin/ directory.
1. Choose a build directory. Ideally separate from the source directory created earlier so that the source can be used again.
	1. Example: `mkdir /tmp/you/mpich-3.4`
	1. Then, navigate into that directory. `cd /tmp/you/mpich-3.4/`
1. Choose configure options as shown in the installation guide.
1. Run the configure script in the source directory (where the tar file was extracted to).
	`/home/you/mpich-3.4/configure -prefix=/home/you/mpich-install 2>&1 | c.txt`
1. Build MPICH with the make command `make 2>&1 | tee m.txt`
1. Install the MPICH commands `make install 2>&1 | mi.txt`
1. 


# Installing mpi4py
1. Ensure Python version 2.7 or 3.3 and above
1. Package managers:
	1. pip install mpi4py
	1. conda install mpi4py