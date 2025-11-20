# 1DREAM PURE CPP LAAT INSTALLATION

To create a executable LAAT file, use the following command in the terminal while you are in this folder:

```sh
make
```

After that, you will obtain several object files, and the 'LAAT.exe' which is your executable file.

Then, run the test.py file to test you LAAT.


```sh
python3 test.py
```

If in the shell (terminal) you see "CODE FINALIZED SUCCESSFULLY" then your LAAT is working!!!. Moreover, you will also find a new file in the Output folder

# USE OF LAAT.exe

Your LAAT.exe is an executable file, and you will need an aditional parameter to call it correctly. This paramter corresponds to the 
'input_file.ini' where you will put all the user paramters to be used. This file must have an extension .ini
You can use the "template_input_file.ini" file in this folder as example.

As an example you can put in your terminal the following command line:

```sh
./LAAT.exe template_input_file.ini
```
