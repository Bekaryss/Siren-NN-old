from MidiUtils import MidiUtils
from Newron import Newron
from sys import stdin
import glob
import os



midiUtil = MidiUtils('midi_songs/*.mid', 30, 100, 160, 0.25, 'output.mid')
newron = Newron(256, 0.1, 512, 'sigmoid', 'binary_crossentropy', "rmsprop", 100, 64)


def train():
    input_network, output_network = midiUtil.preprocessing()
    model = newron.create_model(input_network, '')
    newron.train_model(model, input_network, output_network)

def predict():
    list_of_files = glob.glob('weights_files/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    input_network, output_network = midiUtil.get_predict_midi('predict_song/*')
    model = newron.create_model(input_network, latest_file)
    preditin_matrix = newron.predict(model, input_network, 900)
    midiUtil.postprocessing(preditin_matrix)


if __name__ == '__main__':
    print("Enter 0 for training")
    x = stdin.read(1)
    if(int(x) == 0):
        train()
    else:
        predict()
