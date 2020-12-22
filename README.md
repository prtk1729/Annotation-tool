# Annotation-Tool
A tool to annotate images in a large dataset efficiently by incorporating Machine Learning techniques.


# Project Goals:
* Fast Human Annotation
* Incorporating Active Learning
* File I/O consistency

# Setup (Done only the first time)
* Download Miniconda based on your Operating System(say you have 'Windows' click "Windows Installers") from here --> https://docs.conda.io/en/latest/miniconda.html
* Once downloaded open the exectuable File 'Miniconda3-latest-Windows-x86_64'. And follow the usual installation process.
* After the installation gets completed open command-prompt and type conda --version. If you get a prompt saying: "conda 4.9.2", you have correctly installed.
* Type in command prompt: 
     $ conda create -y -n at37 python=3.7
* Unzip the file 'anaconda_test.zip' that's attached in the mail, that would usually get downloaded to your Downloads folder.
* After unzipping, we end up with an anaconda_test folder.
* On opening the anaconda_test folder we have another 'anaconda_test' and '__MACOSX' folder under the same directory level. Click 'Shift+Right Click' this anaconda_test and click 'Open command window here' which open the command prompt.
* Type 'conda activate at37' and Enter.
* pip install -r requirements.txt
* Type 'python main.py --is_os_win 1' for Windows users and for non-Windows-users type 'python main.py --is_os_win 0'.
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.

# Running everytime after first time
* Open the folder containing the anaconda_zip file i.e ~/Downloads/anaconda_test.
* Click 'Shift+Right Click' this anaconda_test and click 'Open command window here' which open the command prompt.
* Type 'conda activate at37' and Enter.
* Type 'python main.py --is_os_win 1' for Windows users and for non-Windows-users type 'python main.py --is_os_win 0'.
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.

