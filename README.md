# Annotation-Tool
A tool to annotate images in a large dataset efficiently by incorporating Machine Learning techniques.


# Project Goals:
* Fast Human Annotation
* Incorporating Active Learning
* File I/O consistency

# Running the tool after cloning the repo (Refer, only during testing phase with Girish sir and Garima):
* Create a new folder <ann_tool> (say).
* Clone the repo: "git clone https://github.com/prtk1729/Annotation-Tool.git" inside the above created folder.
* Once the repo is cloned navigate to - <ann_tool>/Annotation-Tool/anaconda/anaconda
* Activate a venv/ a conda environment with (python 3.6 or higher) and type "pip install -r requirements.txt".Alternatively, We could simply type this command if we have pip installed with (python 3.6 or higher) without creating a new environment.
* Once the necessary dependencies are installed we could type "python/python3 main.py --is_os_win 1 --initials pp. Note, we set the --is_os_win to '1'(if running on Windows OS) else use --is_os_win 0 (for Non-Win OS) and set the --initials flag to 'pp' (If the name of the Annotator is "Prateek Pani" say).
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.
* The browser tab will show only 3 buttons 'Next', 'Save', 'Export' and nothing else.
* Once the annotator is ready to annotate he/she can go ahead and click 'Next'. 
* The images are displayed with placeholder-values on the radio-values as predicted by the Acquisition Function, the night before.
* Once the annotator annotates all the images in a particular batch. he/she can click 'Save' then 'Next'.
* Continue with the above two steps until you want to end the session.
* In the last batch, click the 'Export' button to save all the annotations made by the annotator in a file (StatsIO/<initials>/<today's_date>/mnist_uptil_today_out_files.json). And the time taken between every two consecutive 'Next' clicks in the file (StatsIO/<initials>/<today's_date>/time_logs.json).
* The annotator need to just copy the entire folder (StatsIO/<initials>/<today's_date>  for e.g StatsIO/pp/26_12_2020) after he/she is done with the annotation process and paste in the SharePoint after navigating to /ML for Med/Annotation_Tool/<gv or gn or pp> or Drag and drop here. 



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

