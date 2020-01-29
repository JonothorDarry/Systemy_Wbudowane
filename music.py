import cv2 as cv
from time import sleep
from mingus.containers import Note, Bar, Track
from mingus.midi import fluidsynth
from FindSymbols import process_image

def grab_image():
    device_id = 0 # /dev/video4
    video_capture = cv.VideoCapture(device_id)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    #video_capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    if not video_capture.isOpened():
        print("Camera not accessible")

    _, image = video_capture.read()
    return image

# cv.imwrite('image.png', grab_image())
def parse_notes(notes):
    track = Track()
    for note in notes:
        if not track.add_notes(note.pitch, note.duration):
            track.add_bar(Bar())
            track.add_notes(note.pitch, note.duration)
    return track

if __name__ == '__main__':
    print('Taking photo of notes...')
    image = grab_image()
    cv.imwrite('captured-image.png', image)
    image = cv.imread('img-sample.jpg')

    print('Processing image...')
    notes = process_image(image)
    track = parse_notes(notes)

    print('Playing music...', flush=True)
    soundfont = 'GeneralUser_GS_1.442-MuseScore/GeneralUser GS MuseScore v1.442.sf2'
    fluidsynth.init(soundfont, 'alsa')

    fluidsynth.play_Track(track)
    print('Program ended')

