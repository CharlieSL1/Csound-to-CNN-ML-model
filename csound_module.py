import ctcsound # type: ignore
import os
import shutil
import time
import sys
import importlib

def reload_module():
    if 'csound_module' in sys.modules:
        importlib.reload(sys.modules['csound_module'])

class Csound_instrument:
    def __init__(self, sr=48000, ksmps=32, nchnls=2, dbfs=1):
        self.sr = sr
        self.ksmps = ksmps
        self.nchnls = nchnls
        self.dbfs = dbfs
        
    def csound_instrument_score(self, instr_number, frequency, duration=1, start_time=0, save_file=None):
        cs = ctcsound.Csound()
        temp_file = f'_temp_{instr_number}.wav'
        if save_file:
            cs.setOption(f'-o {temp_file}')
            cs.setOption('-W')
        
        # Orchestra
        orc = f'''
        sr = {self.sr}
        ksmps = {self.ksmps}
        nchnls = {self.nchnls}
        0dbfs = {self.dbfs}
        
        instr {instr_number}
            ifreq = p4
            aout oscili 0.5, ifreq
            outs aout, aout
        endin
        '''
        
        # Score
        sco = f'i{instr_number} {start_time} {duration} {frequency}'
        
        cs.compileOrc(orc)
        cs.readScore(sco)
        cs.start()
        while cs.performKsmps() == 0:
            pass
        cs.stop()
        cs.cleanup()
        

        if save_file:
            time.sleep(0.1)
            temp_with_space = ' ' + temp_file
            actual_file = temp_with_space if os.path.exists(temp_with_space) else temp_file
            
            if os.path.exists(actual_file):
                target_dir = os.path.dirname(save_file)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                shutil.move(actual_file, save_file)


