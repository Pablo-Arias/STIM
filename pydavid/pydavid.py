#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 5 16:57:12 2018

@author: leehooni
"""

import threading

from pyosc import OSC

# pyDAVID
#

class pydavid():

    address = []

    global user_callback
    def user_callback(path, tags, args, source):
        # which user will be determined by path:
        # we just throw away all slashes and join together what's left
        user = ''.join(path.split("/"))
        # tags will contain 'fff'
        # args is a OSCMessage with data
        # source is where the message came from (in case you need to reply)
        print ("Now do something with", user,args[2],args[0],1-args[1])


    def __init__(self,address):
        self.address = address
        self.client = OSC.OSCClient()
        self.server = OSC.OSCServer((address, 5681))
        self.server.addDefaultHandlers()

        self.server.addMsgHandler( "/user/1", user_callback )
        self.server.addMsgHandler( "/user/2", user_callback )
        self.server.addMsgHandler( "/user/3", user_callback )
        self.server.addMsgHandler( "/user/4", user_callback )
        server = self.server
        self.servert = threading.Thread(target=server.serve_forever)

    def connect(self):
        self.client.connect((self.address, 5678))
        print 'Testing OSC connection...'
        oscmsg = OSC.OSCMessage()
        oscmsg.append('pyDAVID is connected')
        oscmsg.setAddress('/print')

        servert = self.servert
        servert.daemon = True

        servert.start()
        print "Starting OSCServer. Use ctrl-C to quit."

        self.client.send(oscmsg)
        print oscmsg

    def disconnect(self):
#        self.servert.exit()
#        self.server.shutdown()
        self.server.close()
 #       self.servert.join()

    def ping(self):
        oscmsg = OSC.OSCMessage()
        oscmsg.append('ping')
        oscmsg.setAddress('/ping')

        self.client.send(oscmsg)

# MICROPHONE

    def MicOnOff(self,value):
        dbundle = OSC.OSCBundle()
        if (value==0)+(value==1):
            dbundle.append({'addr':"/miconoff", 'args':value})
            client.send(dbundle)
        else:
            raise TypeError("MicOnOff : 0 or 1 expected")

    def MicPreset(self,value):

        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':1})
        dbundle.append({'addr':"/preset", 'args':value})

        self.client.send(dbundle)

    def MicRamp(self,preset=1, hold_time=0, ramp_time=0):

        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':1})
        dbundle.append({'addr':"/preset", 'args':preset})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})

        self.client.send(dbundle)


    def MicPitchShift(self, sfrecname, pitchshift=0, hold_time=0, ramp_time=0, marker_name = [], sfolderrecname = [] ):
        dbundle = OSC.OSCBundle()
        if marker_name != []:
            dbundle.append({'addr':"/recsync", 'args':marker_name})

        dbundle.append({'addr':"/miconoff", 'args':1})
        dbundle.append({'addr':"/pitch", 'args':pitchshift})
        dbundle.append({'addr':"/micrecname", 'args':sfrecname})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})

        if sfolderrecname != []:
            dbundle.append({'addr':"/sfolderrecname", 'args':sfolderrecname})

        dbundle.append({'addr':"/sfrec", 'args':1})

        self.client.send(dbundle)

    def StoreMarkers(self, marker_filename = []):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/recsync-store", 'args':marker_filename})

        self.client.send(dbundle)


    def MicRecord(self, sfrecname, preset=1, hold_time=0, ramp_time=0, sfolderrecname = [] ):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':1})
        dbundle.append({'addr':"/preset", 'args':preset})
        dbundle.append({'addr':"/micrecname", 'args':sfrecname})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})
        #dbundle.append({'addr':"/record", 'args': 1})

        if sfolderrecname != []:
            dbundle.append({'addr':"/sfolderrecname", 'args':sfolderrecname})

        dbundle.append({'addr':"/sfrec", 'args':1})
        self.client.send(dbundle)

    def StopMicRecord(self):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/preset", 'args':1})
        dbundle.append({'addr':"/automation", 'args':[0,0,0]})
        dbundle.append({'addr':"/stoprecord", 'args': [0]})

        self.client.send(dbundle)

# SOUND FILE

    def SfPlay(self,sfplayname = [] ):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/sfplayname", 'args':sfplayname})
        dbundle.append({'addr':"/sfplay", 'args':1})

        self.client.send(dbundle)

    def SfPreset(self,sfplayname,value):

        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/preset", 'args':value})
        dbundle.append({'addr':"/sfplayname", 'args':sfplayname})
        dbundle.append({'addr':"/sfplay", 'args':1})

        self.client.send(dbundle)

    def SfRamp(self,sfplayname,preset=1, hold_time=0, ramp_time=0):

        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/preset", 'args':preset})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})
        dbundle.append({'addr':"/sfplayname", 'args':sfplayname})
        dbundle.append({'addr':"/sfplay", 'args':1})

        self.client.send(dbundle)

    def SfRecord(self,sfplayname, preset=1, hold_time=0, ramp_time=0, sfolderrecname = [] ):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/sfrec", 'args':0})
        dbundle.append({'addr':"/sfplayname", 'args':sfplayname})
        dbundle.append({'addr':"/preset", 'args':preset})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})
        #dbundle.append({'addr':"/record", 'args': 1})
        if sfolderrecname != []:
            dbundle.append({'addr':"/sfolderrecname", 'args':sfolderrecname})
            dbundle.append({'addr':"/sfrec", 'args':1})
        else:
            dbundle.append({'addr':"/sfrec", 'args':1})

        self.client.send(dbundle)

    def SfPitchShiftRecord(self,sfplayname,pitchshift=0, hold_time=0, ramp_time=0, marker_name = [], sfolderrecname = [] ):

        dbundle = OSC.OSCBundle()
        if marker_name != []:
            dbundle.append({'addr':"/recsync", 'args':marker_name})

        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/sfrec", 'args':0})
        dbundle.append({'addr':"/pitch", 'args':pitchshift})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})
        dbundle.append({'addr':"/sfplayname", 'args':sfplayname})
        #dbundle.append({'addr':"/record", 'args':1})
        if sfolderrecname != []:
           dbundle.append({'addr':"/sfolderrecname", 'args':sfolderrecname})
           dbundle.append({'addr':"/sfrec", 'args':1})
        else:
           dbundle.append({'addr':"/sfrec", 'args':1})

        self.client.send(dbundle)

    def SfRecIter(self,sffoldername, sfolderrecname, preset=1, hold_time=0, ramp_time=0):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        dbundle.append({'addr':"/sffoldername", 'args':sffoldername})
        dbundle.append({'addr':"/automation", 'args':[1,hold_time,ramp_time]})
        dbundle.append({'addr':"/preset", 'args':preset})
        dbundle.append({'addr':"/sfolderrecname", 'args':sfolderrecname})
        dbundle.append({'addr':"/sfrec", 'args':1})

        self.client.send(dbundle)

# SMILE

    def sound_duration(self, soundfile):
        """ DURATION OF ANY WAV FILE
        Input   : path of a wav file (string)
        Returns : duration in seconds (float) """
        import wave
        self.soundfile = soundfile

        self.audio = wave.open(self.soundfile)
        self.sr = self.audio.getframerate()               # Sample rate [Hz]
        self.N = self.audio.getnframes()                  # Number of frames (int) 
        self.duration = round(float(self.N)/float(self.sr),2)  # Duration (float) [s]
        return self.duration

    def get_file_list(self, file_path):
        """ GET LIST OF FILE NAMES WITHOUT EXTENSION
        Input   : target folder path (string)
        Returns : file name strings without extension (list) """
        import os
        self.file_path = file_path
        self.file_list = os.listdir(self.file_path)
        if '.DS_Store' in self.file_list: self.file_list.remove('.DS_Store') # for Mac users
        # Remove item in list if it is a directory and remove extension
        self.file_list_names = []
        for self.file_item in self.file_list:
            if os.path.isdir(os.path.join(self.file_path, self.file_item)):
                self.file_list.remove(self.file_item)
            self.file_item, _ = self.file_item.split('.')
            self.file_list_names.append(self.file_item)
        return self.file_list_names

    def ZiggyControl(self, alpha = 1, harm = 0.3, winsize = 1, anawinsize = 1, sfopenpath = '', sfrecname = 'pyDAVID_ziggy', sfplaylistnum = 1, sfrecpath='', warp_freq=[0,0,10000,10000]):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/smile", 'args': ["/alpha",      alpha      ]})
        dbundle.append({'addr':"/smile", 'args': ["/harm",       harm       ]})
        dbundle.append({'addr':"/smile", 'args': ["/winsize",    winsize    ]})
        dbundle.append({'addr':"/smile", 'args': ["/anawinsize", anawinsize ]})
        dbundle.append({'addr':"/smile", 'args': ["/sfplaylistnum", sfplaylistnum ]})
        dbundle.append({'addr':"/smile", 'args': ["/sfrecname", sfrecname ]})
        dbundle.append({'addr':"/smile", 'args': ["/warp_freq", warp_freq ]})
        if sfopenpath!='':
            dbundle.append({'addr':"/smile", 'args': ["/sfopenpath", sfopenpath ]})
        if sfrecpath!='':
            dbundle.append({'addr':"/smile", 'args': ["/sfrecpath", sfrecpath ]})
        self.client.send(dbundle)

    def ZiggyPhoneControl(self, instance=3, alpha = 1, harm = 0.3, winsize = 1, anawinsize = 1, sfopenpath = '', sfrecname = 'pyDAVID_ziggy', sfplaylistnum = 1, sfrecpath=''):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/smile", 'args': ["/inst",      instance    ]})
        dbundle.append({'addr':"/smile", 'args': ["/alpha",      alpha      ]})
        dbundle.append({'addr':"/smile", 'args': ["/harm",       harm       ]})
        dbundle.append({'addr':"/smile", 'args': ["/winsize",    winsize    ]})
        # dbundle.append({'addr':"/smile", 'args': ["/anawinsize", anawinsize ]})
        # dbundle.append({'addr':"/smile", 'args': ["/sfplaylistnum", sfplaylistnum ]})
        # dbundle.append({'addr':"/smile", 'args': ["/sfrecname", sfrecname ]})
        # if sfopenpath!='':
        #     dbundle.append({'addr':"/smile", 'args': ["/sfopenpath", sfopenpath ]})
        # if sfrecpath!='':
        #     dbundle.append({'addr':"/smile", 'args': ["/sfrecpath", sfrecpath ]})
        self.client.send(dbundle)

    def ZigSfRecIter(self):
        dbundle = OSC.OSCBundle()
        dbundle.append({'addr':"/miconoff", 'args':0})
        # dbundle.append({'addr':"/sffoldername", 'args':sffoldername})
        dbundle.append({'addr':"/sfrec", 'args':1})


        self.client.send(dbundle)