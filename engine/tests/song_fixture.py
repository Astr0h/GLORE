import numpy as np
SR, BPM = 44100, 120
BEAT = 60/BPM; BAR = 4*BEAT
def tone(f, dur, amp=0.3, vib=True):
    t = np.arange(int(dur*SR))/SR
    ph = 2*np.pi*f*t + (0.25*np.sin(2*np.pi*5.5*t) if vib else 0)
    env = np.minimum(1, t*25)*np.minimum(1, (t[-1]-t)*25)
    return (sum(np.sin(k*ph)/k for k in range(1,7))*amp*env).astype(np.float32)
m2f = lambda m: 440*2**((m-69)/12)
# verse: busy short notes, quieter; chorus: long held notes (A minor), louder, same each time
VERSE = [(64,.5),(62,.5),(60,.5),(62,.5),(64,1),(None,1)]*2
CHORUS = [(69,2),(67,1),(None,1),(72,2),(71,1),(None,1)]
form = ["V","V","C","C","V","V","C","C","B","C","C"]   # each letter = 2 bars... V/C patterns are 8 beats
def section(kind):
    pat = VERSE if kind=="V" else CHORUS if kind=="C" else [(None,8)]
    amp = 0.18 if kind=="V" else 0.4
    out = []
    for m, d in pat:
        out.append(np.zeros(int(d*BEAT*SR),np.float32) if m is None else tone(m2f(m), d*BEAT, amp))
    x = np.concatenate(out)
    n = int(8*BEAT*SR); return np.pad(x,(0,max(0,n-len(x))))[:n]
voc = np.concatenate([section(k) for k in form])
n = len(voc); bars = int(n/SR/BAR)
# harmony: Am F C G one per bar, pads
prog = [(57,60,64),(53,57,60),(48,52,55),(55,59,62)]
harm = np.zeros(n, np.float32)
for b in range(bars):
    s=int(b*BAR*SR); chord = prog[b%4]
    for m in chord: harm[s:s+int(BAR*SR)] += tone(m2f(m), BAR, 0.12, vib=False)[:max(0,min(int(BAR*SR), n-s))]
chorus_bars = [i*2+j for i,k in enumerate(form) if k=="C" for j in (0,1)]
