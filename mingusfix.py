from time import sleep
from mingus.containers import Note, Bar, Track, Composition
from mingus.midi import fluidsynth

class Player(fluidsynth.FluidSynthSequencer):
    def play_Bars(self, bars, channels, bpm=120):
        """Play several bars (a list of Bar objects) at the same time.
        A list of channels should also be provided. The tempo can be changed
        by providing one or more of the NoteContainers with a bpm argument.
        """
        self.notify_listeners(
            self.MSG_PLAY_BARS, {"bars": bars, "channels": channels, "bpm": bpm}
        )
        qn_length = 60.0 / bpm  # length of a quarter note
        tick = 0.0  # place in beat from 0.0 to bar.length
        cur = [0] * len(bars)  # keeps the index of the NoteContainer under
        # investigation in each of the bars
        playing = []  # The NoteContainers being played.

        while tick < bars[0].length:
            # Prepare a and play a list of NoteContainers that are ready for it.
            # The list `playing_new` holds both the duration and the
            # NoteContainer.
            playing_new = []
            for (n, x) in enumerate(cur):
                if x < len(bars[n]):
                    (start_tick, note_length, nc) = bars[n][x]
                    if start_tick <= tick:
                        self.play_NoteContainer(nc, channels[n])
                        playing_new.append([note_length, n])
                        playing.append([note_length, nc, channels[n], n])
                        cur[n] += 1

                        # Change the length of a quarter note if the NoteContainer
                        # has a bpm attribute
                        if hasattr(nc, "bpm"):
                            bpm = nc.bpm
                            qn_length = 60.0 / bpm

            # Sort the list and sleep for the shortest duration
            if len(playing_new) != 0:
                playing_new.sort()
                shortest = playing_new[-1][0]
                ms = qn_length * (4.0 / shortest)
                self.sleep(ms)
                self.notify_listeners(self.MSG_SLEEP, {"s": ms})
            else:
                # If somehow, playing_new doesn't contain any notes (something
                # that shouldn't happen when the bar was filled properly), we
                # make sure that at least the notes that are still playing get
                # handled correctly.
                if len(playing) != 0:
                    playing.sort()
                    shortest = playing[-1][0]
                    ms = qn_length * (4.0 / shortest)
                    self.sleep(ms)
                    self.notify_listeners(self.MSG_SLEEP, {"s": ms})
                else:
                    # warning: this could lead to some strange behaviour. OTOH.
                    # Leaving gaps is not the way Bar works. should we do an
                    # integrity check on bars first?
                    return {}

            # Add shortest interval to tick
            tick += 1.0 / shortest

            # This final piece adjusts the duration in `playing` and checks if a
            # NoteContainer should be stopped.
            new_playing = []
            for (length, nc, chan, n) in playing:
                duration = 1.0 / length - 1.0 / shortest
                if duration >= 0.00001:
                    new_playing.append([1.0 / duration, nc, chan, n])
                else:
                    self.stop_NoteContainer(nc, chan)
                    # if cur[n] < len(bars[n]) - 1:
                    #    cur[n] += 1
            playing = new_playing

        for p in playing:
            self.stop_NoteContainer(p[1], p[2])
            playing.remove(p)
        return {"bpm": bpm}

