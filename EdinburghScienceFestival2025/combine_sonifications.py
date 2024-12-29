import sounddevice as sd
from pathlib import Path
import ffmpeg as ff
import wavio as wav
import tempfile
import strauss
import os


def combine_save_sonifications(fname, soni1, soni2, ffmpeg_output=False, master_volume=1.):
        """ Save two rendered strauss sonifications as a single, multi-channel wav file, 
        defaults to the sampling rate of the first sonification passed in
         that you want to save out to
          soni1 (:obj:) first rendered sonification
          soni2 (:obj) second rendered sonification
          ffmpeg_output (:obj:`bool`) If True, print :obj:`ffmpeg`
            output to screen 
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1

        Todo:

        Args:
          fname (:obj:`str`) Filename or filepath
          * Either find a way to avoid the need to unscramble channle
        	order, or find alternative to save wav files
        """
        # setup list to house wav stream data 
        inputs = [None]*len(soni1.out_channels)
        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(soni1.out_channels)):
            vmax = max(
                abs((soni1.out_channels[str(c)].values + soni2.out_channels[str(c)].values).max()),
                abs((soni1.out_channels[str(c)].values + soni2.out_channels[str(c)].values).min()),
                vmax
            ) / master_volume
        print("Creating temporary .wav files...")

        # combine caption + sonification streams at display time
        for c in range(len(soni1.out_channels)):
            tempfname = f'.TEMP_{c}.wav'
            print(tempfname)
            soni1.out_channels[str(c)].values += soni2.out_channels[str(c)].values
            if len(soni1.caption_channels[str(c)].values) > 0:
                print('adding caption 1')
                soni1.out_channels[str(c)].values += soni1.caption_channels[str(c)].values
            if len(soni2.caption_channels[str(c)].values) > 0:
                print('adding caption 2')
                soni1.out_channels[str(c)].values += soni2.caption_channels[str(c)].values
            wav.write(tempfname, 
                      soni1.out_channels[str(c)].values,
                      soni1.samprate, 
                      scale = vmax,
                      sampwidth=3)
            inputs[soni1.channels.forder[c]] = ff.input(tempfname)
            
        if ffmpeg_output == False:
            ffmpeg_quiet = True
        else:
            ffmpeg_quiet = False
        print("Joining temporary .wav files...")
        (
            ff.filter(inputs, 'join', inputs=len(inputs), channel_layout=soni1.channels.setup)
            .output(fname)
            .overwrite_output()
            .run(quiet=ffmpeg_quiet)
        )
        
        print("Cleaning up...")
        for c in range(len(soni1.out_channels)):
            tempname = f'.TEMP_{c}.wav'
            if os.path.isfile(tempname):
                os.remove(tempname)
            
        print("Saved to:", fname)