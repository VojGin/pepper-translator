*# coding: utf-8
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import pygame

# record czech or english speech
rec = sr.Recognizer()
with sr.Microphone() as source:
	print("Speak:")
	audio = rec.listen(source)
	recognized = rec.recognize_google(audio, language="cs-CZ")
	print("recognized: " + recognized.encode('utf-8').strip())

# translate to other language
translator = Translator()
translation = translator.translate(recognized, src="cs", dest='en')
translation = translation.text
print(translation)

# say the translation out loud
tts = gTTS(translation, lang="en")
tts.save('say.mp3')

# on Pepper can be replaced by playing the mp3 file through choregraphe box
pygame.mixer.init()
pygame.mixer.music.load("say.mp3")
pygame.mixer.music.play()



''' 
Project info

sources:
https://github.com/Uberi/speech_recognition
(pip install SpeechRecognition)
https://github.com/pndurette/gTTS
(pip install gTTs)
https://github.com/BoseCorp/py-googletrans.git
(pip install googletrans)

Installing libraries on robot (clues):
https://stackoverflow.com/questions/49339754/import-a-local-library-in-choregraphe
or:
https://github.com/pepperhacking/robot-jumpstarter
http://fileadmin.cs.lth.se/robot/nao/doc/software/choregraphe/objects/python_script.html
https://community.ald.softbankrobotics.com/ja/node/10953
https://community.ald.softbankrobotics.com/ja/node/11335


Note:
in case gTTs or googletrans throw the following error: 'NoneType' object has no attribute 'group' , it is a recent bug on google side and needs to be fixed by manual edit of the library. 
see https://github.com/pndurette/gTTS/issues/137, https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group

Connecting to robot:
connect to wifi CIIRC, then in terminal run:
ssh nao@10.37.1.227 
password:nao
'''
