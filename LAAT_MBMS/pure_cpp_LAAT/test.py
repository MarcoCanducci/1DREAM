#Minimum python version = 3.5

import sys

if sys.version_info < (3, 5):
    sys.exit("Please use Python 3.5+")

import subprocess

your_executable_address = "./LAAT.exe"
your_input_file_address = "./template_input_file.ini"

subprocess.run([your_executable_address , your_input_file_address])