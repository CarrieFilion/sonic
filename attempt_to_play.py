import mingus.core.notes as notes
import mingus.core.keys as keys
from mingus.containers import Note
from mingus.midi import fluidsynth
import time



d_list = ['alsa', 'oss', 'jack',
        'portaudio', 'sndmgr', 'coreaudio', 'Direct Sound', 'dsound',
        'pulseaudio']
for driver in d_list:
    fluidsynth.init("GeneralUser GS 1.471/soundfont.sf2", driver)
    print('trying driver:', driver)
    fluidsynth.play_Note(Note("C-5"))
    time.sleep(2.0)
    fluidsynth.stop_Note(Note("C-5"), 1)