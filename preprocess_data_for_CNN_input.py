import scipy
import os as os
from os.path import isfile, join
import pathlib
import numpy as np
import json
from PIL import Image
# import audio_metadata


from koogu import prepare


def main():
    audio_path = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/LongWavs/untitled folder')
    # audio_path = pathlib.Path(input('Where is the raw audio data?'))
    detections_path = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/JoinedSelectionTables/untitled folder')
    # detections_path = pathlib.Path(input('Where is the csv with the detections?'))
    output_path = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Specs_SequencedClips')
    # output_path = pathlib.Path(input('Where should we store the spectrograms?'))
    output_clip_path = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Clips_Sequenced')
    # output_clip_path = pathlib.Path(input('Where should we store the clips?'))

    #onlyaudio = [f for f in sorted(os.listdir(audio_path)) if isfile(join(audio_path, f))]
    included_extensions = ['wav']
    onlyaudio = [fn for fn in os.listdir(audio_path)
                   if any(fn.endswith(ext) for ext in included_extensions)]
    #onlyselects = [f for f in sorted(os.listdir(detections_path)) if isfile(join(detections_path, f))]
    included_extensions = ['txt']
    onlyselects = [fn for fn in os.listdir(detections_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    audio_annot_list = list(zip(onlyaudio, onlyselects))

    # Settings for handling raw audio
    audio_settings = {
        'clip_length': 15.0,
        'clip_advance': 2.5,
        'desired_fs': 250
    }
    # Settings for converting audio to a time-frequency representation
    spec_settings = {
        'win_len': 256,
        'win_overlap': 250,
        'nfft': 3570,
        'bandwidth_clip': [5, 124]
    }

    # Convert audio files into prepared data
    #clip_counts = prepare.from_selection_table_map(
    #    audio_settings,
    #    audio_annot_list,
    #    audio_path, detections_path,
    #    output_root=output_clip_path,
    #    ignore_zero_annot_files=0,
    #    negative_class_label='Noise',
    #    attempt_salvage = True,
    #    show_progress = True
    #)

    pathlist = []
    #labelspath = []
    for subdir, dirs, files in os.walk(output_clip_path):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".npz") & ~subdir.endswith("Koogu"):
                pathlist.append(filepath)
            #if filepath.endswith("labels.npy"):
             #   labelspath.append(filepath)
            if filepath.endswith("list.json"):
                f = open(filepath)
                label_list = json.load(f)

    #for zipped_wavs, zipped_labels in zip(pathlist, labelspath):
    for zipped_location in pathlist:
        data = np.load(zipped_location)
        wavs = data['clips']
        labels = data['labels']

        #wavs = np.load(zipped_wavs)
        #labels = np.load(zipped_labels)

        #save questionable labels file
        csv_name = '_'.join([zipped_location.split('/')[7].split('.')[0], 'questionable-labels.csv'])
        csv_path = '/'.join([str(output_clip_path), csv_name])
        questionable_labels = np.around(labels[labels.sum(axis=1) != 1],2)
        np.savetxt(csv_path, questionable_labels, delimiter = ',', header = ','.join(label_list[:]), comments = '',fmt="%.2f" )

        # Get Sampling rate of original data
        # wavfile = '/'.join([str(audio_path), onlyaudio[pathlist.index(zipped_wavs)]])
        # metadata = audio_metadata.load(wavfile)
        # input_fs = metadata['streaminfo']['sample_rate']

        for x, wav in enumerate(wavs):
            # Create label for clip
            label = labels[x, :].astype(int)
            CLASSNAME = np.array(label_list)[label == 1]
            foobar = np.full((1, len(CLASSNAME)), False) # create an array full of "False"
            idx = np.random.randint(len(CLASSNAME), size=1) # create a list of randomly picked indices, one for each row
            foobar[range(1), idx] = True # replace "False" by "True" at given indices
            ID = x
            TAPEYEAR = zipped_location.split('/')[7].split('.')[0]
            YEAR = TAPEYEAR[len(TAPEYEAR) - 4:len(TAPEYEAR)]
            TAPENAME = TAPEYEAR[0:len(TAPEYEAR) - 4]
            STARTTIMES = int(x * audio_settings['clip_advance'] * 1000)
            ENDTIMES = int(STARTTIMES + audio_settings['clip_length'] * 1000)

            clip_label = ''.join([str(CLASSNAME[tuple(foobar)][0]), '-'])
            clip_label = '_'.join([clip_label, str(ID), YEAR, TAPENAME, str(STARTTIMES), str(ENDTIMES)])
            clip_label = '.'.join([clip_label, 'wav'])

            # Create label for spectrogram
            try:
                ID = int(sum(labels[0:x+1,np.array(label_list) == CLASSNAME[tuple(foobar)][0]].astype(int)))
            except:
                print('this is the one where it stops')
            spec_label = '_'.join([str(ID), TAPEYEAR, CLASSNAME[tuple(foobar)][0]])
            spec_label = '.'.join([spec_label, 'png'])

            # Filter wav
            sos = scipy.signal.iirfilter(20, spec_settings['bandwidth_clip'], rp=None, rs=None, btype='band',
                                         analog=False, ftype='butter', output='sos', fs=audio_settings['desired_fs'])
            wav = scipy.signal.sosfilt(sos, wav)

            # Save clip
            scipy.io.wavfile.write('/'.join([str(output_clip_path), 'AnimalSpot', clip_label]),
                                   audio_settings['desired_fs'], wav)

            # Create and save spectrogram
            f, t, Sxx = scipy.signal.spectrogram(wav, fs=audio_settings['desired_fs'], window=('hamming'),
                                                 nperseg=spec_settings['win_len'],
                                                 noverlap=spec_settings['win_overlap'], nfft=spec_settings['nfft'],
                                                 detrend=False,
                                                 return_onesided=True, scaling='density', axis=-1, mode='magnitude')
            Sxx = 1 - Sxx
            per = np.percentile(Sxx.flatten(), 98)
            I = (Sxx - Sxx.min()) / (per - Sxx.min())
            I[I > 1] = 1
            im = np.array(I * 255, dtype=np.uint8)
            im = Image.fromarray(np.flipud(im))
            im = im.resize((30, 90), Image.Resampling.LANCZOS)
            im.save('/'.join([str(output_path), spec_label]))


if __name__ == "__main__":
    # Compute acoustic features for each detection
    main()
