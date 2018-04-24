from MidiUtils import MidiUtils
from Newron import Newron

midiUtil = MidiUtils('midi_songs/*.mid', 30, 100, 160, 0.25, 'output.mid')
newron = Newron(256, 0.1, 512, 'sigmoid', 'binary_crossentropy', "rmsprop", 100, 64)


def train():
    input_network, output_network = midiUtil.preprocessing()
    model = newron.create_model(input_network, '')
    newron.train_model(model, input_network, output_network)

def predict():
    input_network, output_network = midiUtil.preprocessing
    model = newron.create_model(input_network, '')
    newron.predict(model, input_network, 900)

if __name__ == '__main__':
    train()
