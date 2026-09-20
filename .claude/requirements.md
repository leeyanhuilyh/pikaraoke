## Features

### Pitch change

#### User story

- User needs to be able to change the pitch of the song
- User able to see current pitch relative to song's base pitch

#### Functional requirements

- Pitch change should be seamless (video continues playing, audio continues from where pitch change occurred)
- Pitch change should take no more than 5 seconds

#### Progress

##### live-pitch-shifting

Live, non-restart pitch changing via ffmpeg's azmq control socket driving an always-on rubberband filter, so a pitch change is a runtime command to the existing ffmpeg process rather than a new one. Required pacing ffmpeg to real time (`-re`) and forcing a 3s keyframe interval, since without them the encoder races ahead of playback and the control socket goes silently unresponsive. Fully working end-to-end (verified with real zmq round-trips against a running stream), but doesn't meet the 5-second requirement: song start and the delay before a pitch change is actually audible both land around 9-25s depending on source, an inherent cost of the HLS buffering needed for reliability. Set aside in favor of pitch-resume-position for speed, but structurally the more robust of the two since it has none of the restart-coordination bugs found on that branch.

##### pitch-resume-position

Restart-based: transpose requeues the current song with the new pitch and resumes ffmpeg from the last playback position, meeting the 5-second requirement (~3s in practice). Debugging surfaced four distinct bugs from the restart itself corrupting client state (a skip-triggered video hide blocking a since-added freeze-frame overlay, two separate stale-position races that made the new stream seek past segments it hadn't generated yet, and an unguarded subtitle-worker teardown that could silently abort the whole swap) - all fixed. Two issues remain open: audio/video sync after a transpose is inconsistent between runs at the same semitone shift (not a fixed offset, so not fixable via the avsync setting), and a mid-song crash observed on at least one song with no confirmed root cause yet. Both are suspected to share the same underlying cause as the fixed bugs - ffmpeg isn't paced to real time on this branch - but that fix was deliberately not applied since it would erode the restart-speed advantage that motivated this approach.

### Vocal track toggle

#### User story

- User should be able to toggle on and off the vocal track

#### Functional requirements

- Vocal track should be separate from the backing track
- Track artifact (both vocal and backing) should be kept to a minimum (at dev's discretion)
- On toggle, both vocal and backing tracks should continue seamlessly
- Track switch should take no more than 5 seconds

#### Progress

###### demucs-vocal-separation

Vendors the Demucs fork as a submodule (`vendor/demucs`, a real uv dependency, so plain `uv run pikaraoke` works) and separates each song once, caching a vocals-removed mp3 under `<songs>/.stems/` keyed by YouTube ID, with a size/mtime fingerprint for staleness and cleanup on song delete; the isolated vocal stem is discarded since the original file already serves as the vocals-on side of the toggle. A `vocal_separation` preference (off by default, `background`, `before_play`) and a `vocal_separation_device` preference (`auto`/`cpu`/`cuda`) control it. Downloads and songs already in the library are both covered: queueing a song starts a background separation, and in `before_play` mode playback waits for it behind a "Separating vocals" message on the splash screen. Verified end to end on the RTX 4080 (5s for a 20s clip on GPU, 12s on CPU) and through the running app. The mp3 is encoded by ffmpeg from demucs' lossless output rather than by demucs' own encoder, which left it 25ms late; measured at zero samples of offset against the original. The toggle is live: pre-rendered audio renditions are keyed by `(semitones, vocals_on)`, so a vocals switch or a key change is an HLS audio-track switch with no restart, and a "Vocals: On/Off" button on the remote uses it (`/vocals/on|off`). Vocals-off tracks are declared in the master playlist whenever separation is enabled but only switchable once the song's track exists; in background mode the button starts disabled and enables itself over the socket when separation finishes. Verified against the running app: the vocals-off rendition matches the karaoke stem (0.9997 correlation), and is sample-aligned with the vocals-on one (0 samples), with a switch taking under a second. Not yet done: songs always start with vocals on, and the two preferences are set in config.ini rather than the settings page.

##### vocal-reduction-filter

Implements vocal reduction via ffmpeg's stereotools center-channel-cancellation filter (mode 10), wired through the same restart-and-resume mechanism as transpose, with a shared `_restart_current_with` helper so toggling one setting preserves the other. Explicitly labelled experimental in the UI: this is phase cancellation, not source separation, so it also attenuates other centered instruments and does nothing for off-center vocals - quality varies by song. Already has a `videoLoadGeneration` guard in splash.js against stale timers/events from a superseded video load, a race class also found (and fixed differently) on pitch-resume-position, but has not received that branch's other restart-related fixes (freeze-frame, stale-position races, subtitle-dispose guard), so likely shares some of the same failure modes. Not tested hands-on this session.

### Lyrics display

#### User story

- User should see lyrics for the song show up on screen at the right point in the track (or slightly before the words are sung)
- User should see both the current line and next line of the song
- Lyrics should show up on the left and right of the screen, similar to how it is in karaoke screens

#### Functional requirements

- Lyrics display should be synced up to when the singer actually starts singing
- If lyrics file doesn't exist in local database, it needs to be fetched and synced up to the track

#### Progress
