# Authors
* Dominika Palivcová <palivdom@fel.cvut.cz>
* Vojtěch Gintner <gintnvoj@fel.cvut.cz>

During course A6M33KSY in 2018 at CTU

# Demo

https://www.youtube.com/watch?v=zZAV2IdTwBw

# Install

Run pip install into user folder directly from OpenNAO OS using ssh

```
pip install --upgrade --user pip SpeechRecognition gTTs googletrans pygame six urllib3
```

Put this into your **.bashrc** file

```
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
```

Don't forget to source your new .bashrc by

```
source .bashrc
```

## Note:
in case gTTs or googletrans throw the following error: 'NoneType' object has no attribute 'group' , it is a recent bug on google side and needs to be fixed by manual edit of the library.

https://github.com/pndurette/gTTS/issues/137
https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group

# Run

*pip install needed before running*

Use folder choreographe_project to import project into choreographe and run it directly on Pepper

# Usage

* Say "From English", pepper then records 3 seconds of audio that is then translated to Czech and read aloud along with text being displayed on Pepper's tablet
* Say "From Czech", pepper then records 3 seconds of audio that is then translated to English and read aloud along with text being displayed on Pepper's tablet
* Say "Stop", to break the loop and end the running program

# How to install and run python libraries in OpenNAO (Pepper OS)

Install python libraries into user home folder using:

```
pip install --user
```

Then link those directly in the python script by:

```
import sys; sys.path.insert(0, "/home/nao/.local/lib/python2.7/site-packages/")
```

Some libraries are unable to install using pip (for example pyaudio). These libraries are dependent on system packages (e.g. portaudio) that cannot be build, because there is no C++ compiler in OpenNAO. The only way around is to run OpenNAO in VirtualBox, install it there and then copy to user folder. More info here:

http://doc.aldebaran.com/1-14/dev/tools/vm-intro.html?highlight=virtualbox
http://doc.aldebaran.com/1-14/dev/tools/vm-setup.html?highlight=virtualbox
http://doc.aldebaran.com/1-14/dev/tools/vm-building-thirdparty.html

https://github.com/GoogleCloudPlatform/python-docs-samples/issues/728